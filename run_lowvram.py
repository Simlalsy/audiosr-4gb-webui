# -*- coding: utf-8 -*-
"""
AudioSR low-VRAM runner (tested target: NVIDIA RTX 3050 Laptop, 4 GB).

How it fits AudioSR into ~4 GB of VRAM
--------------------------------------
* Build the complete pipeline on CPU (fp32) so nothing OOMs during loading.
* Cast the diffusion UNet to float16 and move ONLY the UNet to the GPU.
  (fp32 UNet alone would not fit next to the CUDA context.)
* The two VAE copies (first stage + conditioning stage), the VAE decoder and
  the HiFiGAN vocoder stay on CPU - they only run once per chunk, and their
  weights would eat several hundred MB of VRAM each.
* The UNet forward runs inside ``torch.autocast(float16)``, which makes mixed
  fp32/fp16 arithmetic safe without rewriting the model code.
* Attention was patched to ``torch.nn.functional.scaled_dot_product_attention``
  before the models are instantiated: it uses the fused Flash/mem-efficient
  kernels instead of materialising (b*h, N, N) attention matrices.
* ``latent_diffusion.generate_batch`` is monkey-patched so both the shipped
  short-file path (``audiosr.super_resolution``) and the long-audio chunked
  path (``audiosr.super_resolution_long_audio``) work unchanged.

Usage examples
--------------
python run_lowvram.py -i input.wav                       # short file (<= 10 s)
python run_lowvram.py -i long.mp3 --chunking             # long file, chunked
python run_lowvram.py -i input.wav --ddim_steps 30 --chunk_duration 6
python run_lowvram.py -i input.wav --device cpu          # pure CPU fallback

Environment variables
---------------------
AUDIOSR_CKPT     path of a manually downloaded pytorch_model.bin (skips the
                 Hugging Face download)
AUDIOSR_OFFLINE  1 = fully local: never download weights, fail with a clear
                 message when the checkpoint is missing (used by the
                 "fully local" release package)
"""
import argparse
import os
import sys
import time
import types

# Model weights come from Hugging Face - a CN mirror is used by default.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Some Windows setups have a system-wide proxy that the old requests stack
# trips over; bypass the proxy for the Hugging Face downloads in this process.
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

# Offline mode (used by the "fully local" release package): never touch the
# network for weights. The launcher scripts of that package set
# AUDIOSR_OFFLINE=1 for you; if the checkpoint is missing the model loader
# reports it instead of silently trying to download several GB.
OFFLINE = os.environ.get("AUDIOSR_OFFLINE") == "1"
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ckpt_candidates():
    """Places where a manually downloaded pytorch_model.bin is looked for."""
    return [
        os.path.join(BASE_DIR, "pytorch_model.bin"),  # next to the scripts
        os.path.join(BASE_DIR, "models", "pytorch_model.bin"),  # .\models\
        os.path.join(os.path.dirname(BASE_DIR), "pytorch_model.bin"),  # one level up
    ]


def default_ckpt_path():
    """Best guess for the local checkpoint path.

    Order: $AUDIOSR_CKPT, then the first existing candidate from
    ``ckpt_candidates()``. When nothing exists the first candidate is returned
    so the caller can report it (and then either download, or fail in offline
    mode).
    """
    env = os.environ.get("AUDIOSR_CKPT")
    if env:
        return env
    for path in ckpt_candidates():
        if os.path.exists(path):
            return path
    return ckpt_candidates()[0]

# ---------------------------------------------------------------------------
# Patch attention to SDPA BEFORE the model tree is built.
# ---------------------------------------------------------------------------
import audiosr.latent_diffusion.modules.attention as _att  # noqa: E402
from einops import rearrange as _rearrange  # noqa: E402


def _sdpa_forward(self, x, context=None, mask=None):
    h = self.heads
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)
    q, k, v = map(lambda t: _rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
    # no mask is used in AudioSR inference; scale defaults to 1/sqrt(d_head),
    # identical to the original `sim * self.scale`.
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = _rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


_att.CrossAttention.forward = _sdpa_forward

# Gradient checkpointing is pointless during inference (everything runs under
# torch.no_grad) and the parameter style used for it is fragile across torch
# versions; bypass it in inference to save time.
_orig_btb_forward = _att.BasicTransformerBlock.forward


def _btb_forward(self, x, context=None, mask=None):
    if not torch.is_grad_enabled():
        return self._forward(x, context=context, mask=mask)
    return _orig_btb_forward(self, x, context, mask)


_att.BasicTransformerBlock.forward = _btb_forward

from audiosr import build_model, save_wave, get_time  # noqa: E402
from audiosr.pipeline import super_resolution, super_resolution_long_audio  # noqa: E402


def _to_dev(x, dev):
    return x.to(dev) if torch.is_tensor(x) else x


_SCHEDULE_BUFFERS = [
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
]


def set_gpu_vae(ld, enable, device="cuda:0"):
    """Move both VAE copies (+ HiFiGAN vocoder) between CPU-fp32 and GPU-fp16.

    The VAE weights (~3 GB fp32 total) are halved by fp16 and only then fit
    next to the UNet inside 4 GB of VRAM; encode/decode/vocode become much
    faster than their CPU counterparts.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if enable and dev.type == "cuda":
        ld.first_stage_model.half().to(dev)
        ld.cond_stage_models.half().to(dev)
        ld._gpu_vae = True
        print("[lowvram] VAE/vocoder moved to GPU (fp16)")
    else:
        ld.first_stage_model.float().to("cpu")
        ld.cond_stage_models.float().to("cpu")
        ld._gpu_vae = False
        print("[lowvram] VAE/vocoder on CPU (fp32)")


def build_lowvram_model(model_name="basic", device="cuda:0", ckpt_path=None, gpu_vae=False):
    """Build AudioSR on CPU, then move only the fp16 UNet to `device`."""
    if ckpt_path and os.path.exists(ckpt_path):
        # `audiosr.pipeline.build_model` calls `download_checkpoint(model_name)`
        # unconditionally; override it so a manually downloaded file is used.
        import audiosr.pipeline as _pipeline

        _pipeline.download_checkpoint = lambda model_name="basic": ckpt_path
        print(f"[lowvram] using local checkpoint: {ckpt_path}")
    elif OFFLINE:
        raise SystemExit(
            "[lowvram] 离线模式（AUDIOSR_OFFLINE=1）下没有找到本地模型权重：\n"
            f"           {ckpt_path or '(未指定)'}\n"
            "           请把 pytorch_model.bin 放到本程序同一目录（或 models\\ 子目录），\n"
            "           或设置环境变量 AUDIOSR_CKPT 指向该文件。"
        )
    elif ckpt_path:
        print(
            f"[lowvram] checkpoint not found: {ckpt_path} "
            "-> falling back to the Hugging Face download"
        )
    ld = build_model(model_name=model_name, device="cpu")

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[lowvram] CUDA not available, falling back to CPU.")
        device = "cpu"
    dev = torch.device(device)

    if dev.type == "cuda":
        unet = ld.model.diffusion_model
        unet.half().to(dev)  # ~1.2-1.7 GB instead of ~2.5-3.4 GB fp32
    # the DDIM sampling loop touches these root-level buffers directly
    for name in _SCHEDULE_BUFFERS:
        buf = getattr(ld, name, None)
        if torch.is_tensor(buf):
            setattr(ld, name, buf.to(dev))
    ld.device = dev

    if dev.type == "cuda":
        orig_apply = ld.apply_model

        def apply_fp16(x_noisy, t, cond, *a, **kw):
            with torch.autocast("cuda", dtype=torch.float16):
                out = orig_apply(x_noisy, t, cond, *a, **kw)
            return out.float()

        ld.apply_model = apply_fp16

    def generate_batch_lowvram(
        self,
        batch,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        cpu = torch.device("cpu")
        use_gpu_vae = bool(getattr(self, "_gpu_vae", False)) and torch.cuda.is_available()

        # -- 1) VAE encode + conditioning -----------------------------------
        if use_gpu_vae:
            with torch.autocast("cuda", dtype=torch.float16):
                z, c = self.get_input(
                    batch, self.first_stage_key, unconditional_prob_cfg=0.0
                )
        else:
            old_dev = self.device
            self.device = cpu
            try:
                z, c = self.get_input(
                    batch, self.first_stage_key, unconditional_prob_cfg=0.0
                )
            finally:
                self.device = old_dev

        self.latent_t_size = z.size(-2)
        c = self.filter_useful_cond_dict(c)
        batch_size = z.shape[0] * n_gen
        for cond_key in list(c.keys()):
            if isinstance(c[cond_key], list):
                for i in range(len(c[cond_key])):
                    c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
            elif isinstance(c[cond_key], dict):
                for k in c[cond_key].keys():
                    c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
            else:
                c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)

        device = self.device
        z = z.to(device).float()
        c = {
            k: (_to_dev(v, device).float() if torch.is_tensor(v) else v)
            for k, v in c.items()
        }

        uncond = None
        if unconditional_guidance_scale != 1.0:
            uncond = {}
            for key in self.cond_stage_model_metadata:
                model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                uncond[key] = self.cond_stage_models[
                    model_idx
                ].get_unconditional_condition(batch_size)
                uncond[key] = _to_dev(uncond[key], device).float()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- 2) DDIM sampling on GPU (this is the heavy part) --------------
        t0 = time.time()
        samples, _ = self.sample_log(
            cond=c,
            batch_size=batch_size,
            x_T=x_T,
            ddim=True,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uncond,
            use_plms=use_plms,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[lowvram] DDIM {ddim_steps} steps: {time.time() - t0:.1f}s, "
            f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            if torch.cuda.is_available()
            else f"[lowvram] DDIM {ddim_steps} steps: {time.time() - t0:.1f}s (CPU)"
        )

        # -- 3) decode + vocode ---------------------------------------------
        if use_gpu_vae:
            samples = samples.float()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.autocast("cuda", dtype=torch.float16):
                mel = self.decode_first_stage(samples)
            mel = mel.float()
            if torch.cuda.is_available():
                print(
                    f"[lowvram] peak VRAM (incl. VAE decode): "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                )
            lowpass_mel = _to_dev(batch["lowpass_mel"], mel.device)
            mel = self.mel_replace_ops(mel, lowpass_mel)
            with torch.autocast("cuda", dtype=torch.float16):
                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath="", bs=None, save=False
                )
            waveform = np.asarray(waveform, dtype=np.float32)
        else:
            samples = samples.float().to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mel = self.decode_first_stage(samples)
            mel = self.mel_replace_ops(mel, batch["lowpass_mel"])
            waveform = self.mel_spectrogram_to_waveform(
                mel, savepath="", bs=None, save=False
            )
        waveform_lowpass = batch["waveform_lowpass"]
        waveform = self.postprocessing(waveform, waveform_lowpass)
        max_amp = np.max(np.abs(waveform), axis=-1)
        waveform = 0.5 * waveform / max_amp[..., None]
        mean_amp = np.mean(waveform, axis=-1)[..., None]
        waveform = waveform - mean_amp
        return waveform

    ld.generate_batch = types.MethodType(generate_batch_lowvram, ld)
    ld._gpu_vae = False
    if gpu_vae and dev.type == "cuda":
        set_gpu_vae(ld, True, device)
    return ld


# ---------------------------------------------------------------------------
# stereo & orchestration helpers
# ---------------------------------------------------------------------------
DECODABLE_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def ensure_decodable(path):
    """Convert exotic containers (m4a/aac/wma/...) to wav via ffmpeg (channels kept)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in DECODABLE_EXTS:
        return path
    import shutil as _shutil
    import subprocess as _subprocess

    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"不支持的音频格式 {ext}，且系统里找不到 ffmpeg")
    out = os.path.splitext(path)[0] + "_conv.wav"
    _subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", path, "-c:a", "pcm_f32le", out],
        check=True,
    )
    return out


def is_stereo(path):
    try:
        import soundfile as sf

        return sf.info(path).channels == 2
    except Exception:
        return False


def convert_output(wav_path, fmt):
    """Convert the final wav to mp3/flac/m4a/ogg with ffmpeg; keeps the wav."""
    fmt = (fmt or "wav").lower()
    if fmt == "wav":
        return wav_path
    import shutil as _shutil
    import subprocess as _subprocess

    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法转换输出格式")
    codecs = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
        "flac": ["-c:a", "flac"],
        "m4a": ["-c:a", "aac", "-b:a", "256k"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "8"],
    }
    if fmt not in codecs:
        raise ValueError(f"不支持的输出格式: {fmt}")
    out_path = os.path.splitext(wav_path)[0] + "." + fmt
    _subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", wav_path, *codecs[fmt], out_path],
        check=True,
    )
    print(f"[lowvram] converted -> {out_path}")
    return out_path


def _split_stereo(path, out_dir):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    paths = []
    for ch in range(data.shape[1]):
        p = os.path.join(out_dir, f"{stem}_ch{ch}.wav")
        sf.write(p, data[:, ch], sr, subtype="FLOAT")
        paths.append(p)
    return paths


def _process_file(
    ld, path, ddim_steps, seed, guidance_scale, chunk_duration_s,
    overlap_duration_s, chunking="auto",
):
    """Run AudioSR on ONE mono file; returns a (1, 1, T) tensor."""
    import torchaudio

    try:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
    except Exception:
        duration = None
    if chunking is None:
        chunking = "auto"
    if chunking == "auto":
        use_chunk = duration is not None and duration > 12
    else:
        use_chunk = bool(chunking)
    if use_chunk:
        dur_txt = f"{duration:.1f}s" if duration is not None else "long"
        print(
            f"[lowvram] {os.path.basename(path)}: {dur_txt} -> "
            f"chunked mode ({chunk_duration_s}s / {overlap_duration_s}s overlap)"
        )
        wav = super_resolution_long_audio(
            ld, path, seed=seed, guidance_scale=guidance_scale,
            ddim_steps=ddim_steps, chunk_duration_s=chunk_duration_s,
            overlap_duration_s=overlap_duration_s,
        )
    else:
        wav = super_resolution(
            ld, path, seed=seed, guidance_scale=guidance_scale,
            ddim_steps=ddim_steps, latent_t_per_second=12.8,
        )
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    return wav


def process_and_save(
    ld, in_path, out_dir, name, ddim_steps=50, seed=42, guidance_scale=3.5,
    chunking="auto", chunk_duration_s=10, overlap_duration_s=2, stereo_mode="split",
    out_sr=48000, out_format="wav",
):
    """Full pipeline: optional stereo split -> per-channel AudioSR -> merge back.

    stereo_mode="split" keeps true stereo (L/R processed separately, ~2x time);
    "mono" keeps the legacy behaviour (left channel / downmix).
    The model natively restores to 48 kHz; out_sr != 48000 is a high-quality
    sinc resample of that result (e.g. 44100 for CD, 16000 for speech models).
    out_format != "wav" converts the result with ffmpeg (mp3/flac/m4a/ogg);
    the wav master is always kept next to it.
    Returns the final output path.
    """
    import shutil as _shutil
    import tempfile

    import soundfile as sf

    out_sr = int(max(8000, min(192000, int(out_sr))))
    os.makedirs(out_dir, exist_ok=True)

    if stereo_mode == "split" and is_stereo(in_path):
        data_in, _sr_in = sf.read(in_path, dtype="float32", always_2d=True)
        in_peaks = [float(np.max(np.abs(data_in[:, i]))) + 1e-8 for i in range(2)]
        tmp = tempfile.mkdtemp(prefix="audiosr_st_")
        try:
            ch_paths = _split_stereo(in_path, tmp)
            outs = []
            for ch in ch_paths:
                print(f"[lowvram] processing channel: {os.path.basename(ch)}")
                w = _process_file(
                    ld, ch, ddim_steps, seed, guidance_scale,
                    chunk_duration_s, overlap_duration_s, chunking,
                )
                # the chunked library path returns (1, T) (batch dim squeezed),
                # the single-pass path returns (1, 1, T); flatten both to 1-D
                outs.append(w.detach().cpu().float().numpy().reshape(-1))
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
        # Channel loudness handling differs per library path (the chunked path
        # already restores each chunk to its original peak, the single-pass path
        # normalises to 0.5), so re-normalise BOTH channels deterministically:
        # louder channel -> 0.5, the other keeps the original L/R peak ratio.
        rel = min(max(in_peaks[1] / in_peaks[0], 0.05), 20.0)
        o0 = np.asarray(outs[0], dtype=np.float32)
        o1 = np.asarray(outs[1], dtype=np.float32)
        o0 = o0 / (float(np.max(np.abs(o0))) + 1e-8) * 0.5
        o1 = o1 / (float(np.max(np.abs(o1))) + 1e-8) * 0.5 * rel
        n = min(len(o0), len(o1))
        stereo = np.stack([o0[:n], o1[:n]], axis=1)
        peak = float(np.max(np.abs(stereo))) + 1e-8
        if peak > 0.99:
            stereo = stereo * (0.99 / peak)
        if out_sr != 48000:
            import torchaudio

            t = torch.from_numpy(np.ascontiguousarray(stereo.T))
            stereo = torchaudio.functional.resample(t, 48000, out_sr).numpy().T
        out_path = os.path.join(out_dir, name + ".wav")
        sf.write(out_path, stereo, out_sr, subtype="PCM_16")
        print(f"[lowvram] stereo merged -> {out_path} ({out_sr} Hz)")
        return convert_output(out_path, out_format)

    wav = _process_file(
        ld, in_path, ddim_steps, seed, guidance_scale,
        chunk_duration_s, overlap_duration_s, chunking,
    )
    if out_sr != 48000:
        import torchaudio

        wav = torchaudio.functional.resample(wav, 48000, out_sr)
    save_wave(wav, inputpath=in_path, savepath=out_dir, name=name, samplerate=out_sr)
    return convert_output(os.path.join(out_dir, name + ".wav"), out_format)


def main():
    ap = argparse.ArgumentParser(
        description="AudioSR low-VRAM runner (fp16 UNet on GPU, rest on CPU)"
    )
    ap.add_argument("-i", "--input_audio_file", required=True)
    ap.add_argument("-s", "--save_path", default="./output")
    ap.add_argument("--model_name", default="basic", choices=["basic", "speech"])
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("-gs", "--guidance_scale", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="_AudioSR_Processed_48K")
    ap.add_argument(
        "--out_sr",
        type=int,
        default=48000,
        help="output sample rate in Hz, 8000-192000 (the model natively outputs 48000; higher rates are upsampled containers - 24 kHz is the hard content limit, nothing above it is generated)",
    )
    ap.add_argument(
        "--out_format",
        choices=["wav", "mp3", "flac", "m4a", "ogg"],
        default="wav",
        help="convert the output to this format with ffmpeg (the wav master is kept too)",
    )
    ap.add_argument("--chunking", action="store_true", help="chunk long audio")
    ap.add_argument("--chunk_duration", type=int, default=10)
    ap.add_argument("--overlap_duration", type=int, default=2)
    ap.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="cuda:0 (default) or cpu",
    )
    ap.add_argument(
        "--ckpt_path",
        default=default_ckpt_path(),
        help="path to a manually downloaded pytorch_model.bin; skipped when missing "
        "(override with the AUDIOSR_CKPT environment variable)",
    )
    ap.add_argument(
        "--stereo",
        choices=["split", "mono"],
        default="split",
        help="stereo input handling: 'split' = process L/R separately (true stereo, ~2x time); 'mono' = legacy downmix",
    )
    ap.add_argument(
        "--gpu_vae",
        action="store_true",
        help="run the VAE encoder/decoder + vocoder on the GPU in fp16 (faster, uses more VRAM)",
    )
    args = ap.parse_args()

    if args.out_sr != 48000 and args.suffix == "_AudioSR_Processed_48K":
        args.suffix = f"_AudioSR_Processed_{args.out_sr / 1000:g}K"

    src = ensure_decodable(args.input_audio_file)
    save_path = os.path.join(args.save_path, get_time())
    os.makedirs(save_path, exist_ok=True)

    ld = build_lowvram_model(args.model_name, args.device, args.ckpt_path or None, args.gpu_vae)
    name = os.path.splitext(os.path.basename(src))[0] + args.suffix

    if args.stereo == "split" and is_stereo(src):
        print("[lowvram] stereo input detected -> per-channel processing (true stereo, ~2x time)")

    t0 = time.time()
    out = process_and_save(
        ld, src, save_path, name,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        chunking=True if args.chunking else "auto",
        chunk_duration_s=args.chunk_duration,
        overlap_duration_s=args.overlap_duration,
        stereo_mode=args.stereo,
        out_sr=args.out_sr,
        out_format=args.out_format,
    )
    print(f"[lowvram] total processing time: {time.time() - t0:.1f}s")
    print(f"[lowvram] saved to: {out}")


if __name__ == "__main__":
    main()
