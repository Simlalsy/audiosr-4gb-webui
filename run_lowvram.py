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
import glob
import hashlib
import os
import shutil
import sys
import threading
import time
import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model weights come from Hugging Face - a CN mirror is used by default.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Some Windows setups have a system-wide proxy that the old requests stack
# trips over; bypass the proxy for the Hugging Face downloads in this process.
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
# Keep every download inside the program folder (do NOT fill up the C: drive):
# checkpoint, hub cache and anything else huggingface_hub stores.
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "models"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE_DIR, "models", "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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


# SHA-256 of the official basic checkpoint (haoheliu/audiosr_basic).
BASIC_MODEL_SHA256 = "8a3506b9619ed32435ce2c115604750c7bddb5ad8502be7b1a3131bde878aa01"

DOWNLOAD_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


def download_checkpoint_with_fallback(model_name="basic"):
    """Download the checkpoint, trying the CN mirror first and then the Hub.

    Everything is stored inside ``<program folder>\\models`` (HF_HOME and
    HF_HUB_CACHE are set above, so the C: drive is left alone), and a hard link
    is created next to the scripts so later runs - and the offline mode - find
    the file immediately.
    """
    from huggingface_hub import hf_hub_download

    repo_id = (
        "haoheliu/audiosr_speech" if model_name == "speech" else "haoheliu/audiosr_basic"
    )
    errors = []
    for endpoint in DOWNLOAD_ENDPOINTS:
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            print(
                f"[lowvram] 下载模型权重 {repo_id}/pytorch_model.bin（约 5.75 GB）…\n"
                f"[lowvram] 来源：{endpoint}\n"
                f"[lowvram] 保存到：{os.environ['HF_HOME']}（不会占用 C 盘）"
            )
            path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            print(f"[lowvram] 下载完成：{path}")
            link = os.path.join(BASE_DIR, "pytorch_model.bin")
            if model_name == "basic" and not os.path.exists(link):
                try:
                    os.link(path, link)
                    print(f"[lowvram] 已在程序目录建立硬链接：{link}")
                except OSError as exc:  # not fatal - the cache copy is used
                    print(f"[lowvram] 建立硬链接失败（不影响使用）：{exc}")
            return path
        except Exception as exc:  # noqa: BLE001 - report it and try the next source
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
            print(f"[lowvram] 从 {endpoint} 下载失败，尝试下一个来源…")
    raise SystemExit(
        "[lowvram] 模型权重下载失败：\n  "
        + "\n  ".join(errors)
        + "\n  可手动下载后放到程序目录（与 .cmd 同层）或 models\\ 子目录：\n"
        "    https://hf-mirror.com/haoheliu/audiosr_basic/resolve/main/pytorch_model.bin"
    )


def _file_sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_model_parts(target=None, verify=True):
    """Rebuild ``pytorch_model.bin`` from ``pytorch_model.bin.part*`` slices.

    The "fully offline" release ships the checkpoint as several release assets
    (< 2 GiB each, the GitHub limit). On the first run the slices are merged
    back into one file next to the scripts and verified with SHA-256.
    Returns the merged path, or None when there are no parts to merge.
    """
    target = target or os.path.join(BASE_DIR, "pytorch_model.bin")
    if os.path.exists(target):
        return target
    parts = sorted(glob.glob(target + ".part*"))
    if not parts:
        return None

    print(
        f"[lowvram] 首次运行：正在合并 {len(parts)} 个模型分卷 -> "
        f"{os.path.basename(target)}"
    )
    tmp = target + ".merging"
    try:
        with open(tmp, "wb") as out:
            for i, part in enumerate(parts, 1):
                size_mb = os.path.getsize(part) / 2 ** 20
                print(
                    f"[lowvram]   分卷 {i}/{len(parts)}："
                    f"{os.path.basename(part)}（{size_mb:.0f} MB）"
                )
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out, 1 << 24)
        if verify:
            print("[lowvram] 校验合并结果（SHA-256，约 10 秒）…")
            digest = _file_sha256(tmp)
            if digest != BASIC_MODEL_SHA256:
                raise SystemExit(
                    "[lowvram] 模型分卷合并后校验失败，请重新下载分卷：\n"
                    f"           期望 {BASIC_MODEL_SHA256}\n"
                    f"           实际 {digest}"
                )
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[lowvram] 模型已就绪：{target}")
    print("[lowvram] （下次启动将直接使用该文件，不再合并）")
    return target

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
    if model_name == "basic" and not (ckpt_path and os.path.exists(ckpt_path)):
        # the offline release ships the checkpoint as <2 GiB slices
        merged = merge_model_parts()
        if merged:
            ckpt_path = merged
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
    else:
        print("[lowvram] 未提供本地权重，将自动下载到程序目录的 models\\ 下")

    if not (ckpt_path and os.path.exists(ckpt_path)) and not OFFLINE:
        #下载失败时自动换源（国内镜像 -> 官方站）
        import audiosr.pipeline as _pipeline

        _pipeline.download_checkpoint = download_checkpoint_with_fallback
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
        # `sample_log` forwards **kwargs to DDIMSampler.sample(), which already
        # calls `callback(i)` after every step - that is our progress/cancel
        # checkpoint (one call per chunk, so chunk counting stays exact).
        _progress_enter_chunk()
        step_callback = (
            None
            if use_plms
            else (lambda i, _n=int(ddim_steps): _progress_step(i, _n))
        )
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
            callback=step_callback,
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


# ---------------------------------------------------------------------------
# Progress reporting + cancellation (the CLI ignores both; the Web UI uses them)
#
# `generate_batch` runs exactly once per processed chunk - single pass, library
# chunking, fork chunking and each stereo channel included - so the pipeline
# reports progress in *chunks* and refines it with the DDIM step callback the
# sampler already supports (`DDIMSampler.sample(callback=...)`). The same
# callback is the cancellation checkpoint, which makes "stop" react within one
# DDIM step instead of only between chunks.
# ---------------------------------------------------------------------------
PROGRESS_SR = 48000


class JobCancelled(Exception):
    """Raised inside the pipeline when a stop was requested (Web UI 停止)."""


_progress_hook = None
_cancel_event = threading.Event()
_prog = {"total": 0, "chunk": 0, "step": 0.0, "step_i": 0, "steps": 0, "label": ""}


def set_progress_hook(hook):
    """Install `hook(fraction, text)` (or None) receiving 0..1 progress."""
    global _progress_hook
    _progress_hook = hook


def request_cancel():
    """Ask the running pipeline to stop at its next checkpoint."""
    _cancel_event.set()


def clear_cancel():
    _cancel_event.clear()
    _progress_reset()


def _progress_reset():
    _prog.update(total=0, chunk=0, step=0.0, step_i=0, steps=0, label="")


def cancel_requested():
    return _cancel_event.is_set()


def check_cancel():
    if _cancel_event.is_set():
        raise JobCancelled("已请求停止")


def _progress_text():
    """Short "what is happening now" string, e.g. `声道 1/2 · 块 3/8 · 采样 12/50`."""
    parts = []
    if _prog["label"]:
        parts.append(_prog["label"])
    if _prog["total"] > 1:
        parts.append(f"块 {min(max(_prog['chunk'], 1), _prog['total'])}/{_prog['total']}")
    if _prog["steps"] > 1:
        parts.append(f"采样 {min(max(_prog['step_i'], 1), _prog['steps'])}/{_prog['steps']}")
    return " · ".join(parts)


def _progress_report():
    hook = _progress_hook
    if hook is None or _prog["total"] <= 0:
        return
    done = (_prog["chunk"] - 1 + _prog["step"]) / _prog["total"]
    try:
        hook(min(1.0, max(0.0, done)), _progress_text())
    except Exception:  # a broken hook must never kill a job
        pass


def _progress_start(total_chunks, label=""):
    _progress_reset()
    _prog.update(total=max(int(total_chunks), 0), label=label)
    _progress_report()


def _progress_set_label(label):
    if label != _prog["label"]:
        _prog["label"] = label
        _progress_report()


def _progress_enter_chunk():
    """Called once per chunk, right before its DDIM loop starts."""
    check_cancel()
    _prog["chunk"] += 1
    _prog.update(step=0.0, step_i=0, steps=0)
    _progress_report()


def _progress_step(index, steps):
    """DDIM step callback: `index` is the 0-based index of the finished step."""
    check_cancel()
    steps = max(int(steps), 1)
    _prog["steps"] = steps
    _prog["step_i"] = int(index) + 1
    _prog["step"] = min(1.0, _prog["step_i"] / steps)
    _progress_report()


def _audio_duration(path):
    try:
        import torchaudio

        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        return None


def _chunk_geometry(chunk_duration_s, overlap_duration_s, sr=PROGRESS_SR):
    chunk_samples = max(int(chunk_duration_s * sr), sr // 4)
    overlap_samples = int(min(max(overlap_duration_s, 0), chunk_duration_s - 0.1) * sr)
    if overlap_samples >= chunk_samples:
        overlap_samples = 0
    return chunk_samples, overlap_samples


def _chunk_count(total_samples, chunk_duration_s, overlap_duration_s, fork_style=False):
    """How many chunks a file of `total_samples` will be split into."""
    chunk_samples = max(int(chunk_duration_s * PROGRESS_SR), 1)
    if not fork_style:
        # audiosr.super_resolution_long_audio: range(0, total, chunk - overlap)
        step = chunk_samples - int(overlap_duration_s * PROGRESS_SR)
        return 1 if step <= 0 else max(1, int(np.ceil(total_samples / step)))
    _, overlap_samples = _chunk_geometry(chunk_duration_s, overlap_duration_s)
    step = chunk_samples - overlap_samples
    if total_samples <= chunk_samples:
        return 1
    return 1 + int(np.ceil((total_samples - chunk_samples) / step))


def _planned_chunks(path, chunking, chunk_duration_s, overlap_duration_s, fork_style=False):
    duration = _audio_duration(path)
    if chunking is None:
        chunking = "auto"
    if chunking == "auto":
        use_chunk = duration is not None and duration > 12
    else:
        use_chunk = bool(chunking)
    if not use_chunk or duration is None:
        return 1
    return _chunk_count(
        int(round(duration * PROGRESS_SR)), chunk_duration_s, overlap_duration_s, fork_style
    )


# ---------------------------------------------------------------------------
# Optional synthetic top band ("exciter")
#
# AudioSR cannot put anything above 24 kHz (48 kHz mel + 48 kHz vocoder), so a
# 96/192 kHz export is silent up there. This post-process squares the top
# octave of the output, which folds *synthesised* 2nd-order harmonics into
# 24-43 kHz and makes the high-rate file sound "open". It is NOT recovered
# detail - it is generated content, mixed far below the original level - and it
# is only applied when out_sr > 48000.
# ---------------------------------------------------------------------------
AIR_LEVELS = {"off": 0.0, "light": 0.03, "medium": 0.06, "strong": 0.12}


def _biquad(audio, cutoff, btype, sr, order=4):
    from scipy import signal as _sig

    nyquist = 0.5 * sr
    wn = float(min(max(cutoff, 1.0), nyquist - 1.0)) / nyquist
    sos = _sig.butter(order, wn, btype=btype, output="sos")
    return _sig.sosfiltfilt(sos, audio)


def _synthetic_air(audio, sr, amount=0.06):
    """Mix synthesised harmonics above 24 kHz into a 1-D float signal.

    2nd-order harmonics of the 12-24 kHz octave land in 24-48 kHz, so squaring
    that band fills the top of a high-rate export without aliasing (a 3rd-order
    term would alias, hence 2nd order only). The source octave is extracted with
    a steep filter and gated on its level, so material that is band-limited
    below ~12 kHz is returned completely untouched.
    """
    if amount <= 0 or sr <= 48000 or len(audio) < 4096:
        return audio  # true no-op - the caller's array/dtype is untouched
    audio = np.asarray(audio, dtype=np.float32)
    top = min(24000.0, 0.45 * sr)
    source = _biquad(
        _biquad(audio, 12000.0, "highpass", sr, order=8), top, "lowpass", sr, order=8
    )
    ref = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-12
    src = float(np.sqrt(np.mean(source.astype(np.float64) ** 2))) + 1e-12
    if src < 0.002 * ref:  # top octave is ~54 dB down: nothing worth exciting
        return audio
    harm = _biquad(
        _biquad(source.astype(np.float64) ** 2, 24500.0, "highpass", sr),
        min(46000.0, 0.45 * sr),
        "lowpass",
        sr,
    )
    cur = float(np.sqrt(np.mean(harm ** 2))) + 1e-12
    mixed = audio.astype(np.float64) + harm / cur * ref * float(amount)
    peak = float(np.max(np.abs(mixed))) + 1e-12
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# jarredou AudioSR-Colab-Fork DSP port
# (https://github.com/brahianrosswill/AudioSR-Colab-Fork - the same processing
# that ships in that fork's inference.py, wired into the low-VRAM pipeline):
#   * input_cutoff        band-limit the model input by resampling it to
#                         cutoff*2 Hz instead of low-pass filtering it (v0.5);
#   * multiband_ensemble  keep the original audio below `cutoff - 1000` Hz and
#                         mix in the generated audio above it (v0.3);
#   * loudness_match      match every processed chunk to the integrated
#                         loudness (LUFS) of its input chunk instead of its
#                         peak, which removes the volume jumps between chunks
#                         (v0.5). Needs `pyloudnorm`; without it the previous
#                         peak matching is used.
# ---------------------------------------------------------------------------
FORK_DEFAULT_CUTOFF = 14000


def _fork_cutoff(input_cutoff):
    """Normalise the cutoff argument: 0/None/bogus -> 0 (feature disabled)."""
    try:
        cutoff = int(input_cutoff)
    except (TypeError, ValueError):
        return 0
    return cutoff if cutoff > 2000 else 0


def _match_len(audio, length):
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio


def _load_mono(path, sr=48000):
    """Read the first channel of `path` and resample it to `sr`."""
    import soundfile as sf

    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(data[:, 0], dtype=np.float32)
    if file_sr != sr:
        audio = _resample(audio, file_sr, sr)
    return audio


def _resample(audio, orig_sr, target_sr):
    """High-quality sample-rate conversion (time on the LAST axis).

    `torchaudio.functional.resample` uses a short Hann-windowed sinc: just above
    its cutoff the stop-band is only ~-15 dB, so a 48 -> 96 kHz export folded
    audible images of the 21-24 kHz band into 24-27 kHz (and downsampling folded
    ~-70 dB of aliasing back into the passband). librosa/soxr reject both by
    100+ dB; `soxr` is used when installed, otherwise librosa's own filter.
    """
    if int(orig_sr) == int(target_sr):
        return audio
    import librosa

    res_type = "kaiser_best"
    try:
        import soxr  # noqa: F401  - enables librosa's faster soxr_* presets

        res_type = "soxr_hq"
    except Exception:
        pass
    return librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=int(orig_sr),
        target_sr=int(target_sr),
        res_type=res_type,
    )


def _lr_filter(audio, cutoff, btype, order=10, sr=48000):
    """Zero-phase Butterworth filter; same filter the fork's `lr_filter` uses."""
    from scipy import signal as _sig

    nyquist = 0.5 * sr
    cutoff = float(min(max(cutoff, 1.0), nyquist - 1.0))
    sos = _sig.butter(order // 2, cutoff / nyquist, btype=btype, output="sos")
    return _sig.sosfiltfilt(sos, audio).astype(np.float32)


def _band_limit(path, cutoff, tmp_dir):
    """Write a copy of `path` resampled to `cutoff * 2` Hz (fork v0.5)."""
    import soundfile as sf

    target_sr = int(cutoff) * 2
    audio = _load_mono(path, sr=target_sr)
    out_path = os.path.join(
        tmp_dir, f"{os.path.splitext(os.path.basename(path))[0]}.cutoff{cutoff}.wav"
    )
    sf.write(out_path, audio, target_sr, subtype="FLOAT")
    return out_path


def _multiband_ensemble(original, generated, cutoff, sr=48000):
    """Original audio below the crossover + generated audio above it (fork v0.3)."""
    if len(original) < 512 or len(generated) < 512:
        return _match_len(generated, len(original))
    crossover = min(cutoff - 1000, sr // 2 - 500)
    n = len(original)
    low = _lr_filter(_match_len(original, n), crossover, "lowpass", order=10, sr=sr)
    high = _lr_filter(_match_len(generated, n), crossover, "highpass", order=10, sr=sr)
    high = _lr_filter(high, min(23000, sr // 2 - 500), "lowpass", order=2, sr=sr)
    return (low + high).astype(np.float32)


def _loudness_match(reference, target, sr=48000):
    """Scale `target` to the integrated loudness (LUFS) of `reference`.

    Falls back to peak matching when pyloudnorm is unavailable or the block is
    shorter than its 400 ms loudness window (the fork has no such guard and
    crashes on very short chunks).
    """
    reference = np.asarray(reference, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        ref_lufs = meter.integrated_loudness(reference.astype(np.float64))
        tgt_lufs = meter.integrated_loudness(target.astype(np.float64))
        if not (np.isfinite(ref_lufs) and np.isfinite(tgt_lufs)):
            raise ValueError("non-finite integrated loudness")
        return pyln.normalize.loudness(
            target.astype(np.float64), tgt_lufs, ref_lufs
        ).astype(np.float32)
    except Exception:
        ref_peak = float(np.max(np.abs(reference))) + 1e-8
        tgt_peak = float(np.max(np.abs(target))) + 1e-8
        return (target / tgt_peak * ref_peak).astype(np.float32)


def _chunked_sr_loudness_matched(
    ld, path, ddim_steps, seed, guidance_scale, chunk_duration_s,
    overlap_duration_s, sr=48000,
):
    """Fork-style chunk loop: per-chunk LUFS matching + linear cross-fades.

    Same chunk/overlap geometry as `audiosr.super_resolution_long_audio`, but
    every generated chunk is matched to the loudness of the input chunk it came
    from instead of being peak-normalised (fork v0.5). The linear fade in/out
    ramps sum to 1, so the overlapping regions need no extra normalisation.
    """
    import soundfile as sf
    import tempfile

    audio = _load_mono(path, sr=sr)
    chunk_samples, overlap_samples = _chunk_geometry(chunk_duration_s, overlap_duration_s, sr)
    step = chunk_samples - overlap_samples
    n_chunks = _chunk_count(len(audio), chunk_duration_s, overlap_duration_s, fork_style=True)
    # every chunk is trimmed to its unpadded length, so the overlap-add output
    # is exactly as long as the input (no silent tail from the last padding)
    out = np.zeros(len(audio), dtype=np.float32)

    fade_in = fade_out = None
    if overlap_samples:
        fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
        fade_out = fade_in[::-1].copy()

    tmp_dir = tempfile.mkdtemp(prefix="audiosr_chunk_")
    try:
        for i in range(n_chunks):
            start = i * step
            chunk = audio[start:start + chunk_samples]
            if not len(chunk):
                break
            original_length = len(chunk)
            chunk_path = os.path.join(tmp_dir, f"chunk{i}.wav")
            sf.write(chunk_path, _match_len(chunk, chunk_samples), sr, subtype="FLOAT")
            generated = super_resolution(
                ld, chunk_path, seed=seed, guidance_scale=guidance_scale,
                ddim_steps=ddim_steps, latent_t_per_second=12.8,
            )
            if isinstance(generated, np.ndarray):
                generated = torch.from_numpy(generated)
            generated = (
                generated.detach().cpu().float().numpy().reshape(-1)[:original_length]
            )
            generated = _loudness_match(chunk[:original_length], generated, sr=sr)
            if fade_in is not None:
                n_fade = min(overlap_samples, len(generated))
                if i > 0:
                    generated[:n_fade] *= fade_in[:n_fade]
                if i < n_chunks - 1:
                    generated[-n_fade:] *= fade_out[-n_fade:]
            end = min(start + len(generated), len(out))
            out[start:end] += generated[: end - start]
            print(
                f"[lowvram] chunk {i + 1}/{n_chunks} "
                f"(LUFS matched, {original_length / sr:.2f}s)"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return torch.from_numpy(out).unsqueeze(0)


def _process_file(
    ld, path, ddim_steps, seed, guidance_scale, chunk_duration_s,
    overlap_duration_s, chunking="auto", input_cutoff=0,
    multiband_ensemble=False, loudness_match=False, label="",
):
    """Run AudioSR on ONE mono file; returns a (1, T) tensor."""
    import tempfile

    import torchaudio

    check_cancel()
    _progress_set_label(label)

    cutoff = _fork_cutoff(input_cutoff)
    if multiband_ensemble and not cutoff:
        cutoff = FORK_DEFAULT_CUTOFF

    tmp_dir = None
    model_path = path
    if cutoff:
        tmp_dir = tempfile.mkdtemp(prefix="audiosr_cutoff_")
        model_path = _band_limit(path, cutoff, tmp_dir)
        print(f"[lowvram] input band-limited to {cutoff} Hz (fork input_cutoff)")

    try:
        try:
            info = torchaudio.info(model_path)
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
                f"[lowvram] {os.path.basename(model_path)}: {dur_txt} -> "
                f"chunked mode ({chunk_duration_s}s / {overlap_duration_s}s overlap)"
            )
            if loudness_match:
                wav = _chunked_sr_loudness_matched(
                    ld, model_path, ddim_steps, seed, guidance_scale,
                    chunk_duration_s, overlap_duration_s,
                )
            else:
                wav = super_resolution_long_audio(
                    ld, model_path, seed=seed, guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps, chunk_duration_s=chunk_duration_s,
                    overlap_duration_s=overlap_duration_s,
                )
        else:
            wav = super_resolution(
                ld, model_path, seed=seed, guidance_scale=guidance_scale,
                ddim_steps=ddim_steps, latent_t_per_second=12.8,
            )
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.detach().cpu().float().reshape(-1)
        if loudness_match and not use_chunk:
            reference = _load_mono(model_path)
            n = min(len(wav), len(reference))
            wav = torch.from_numpy(_loudness_match(reference[:n], wav[:n].numpy()))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if multiband_ensemble and cutoff:
        mixed = _multiband_ensemble(_load_mono(path), wav.numpy(), cutoff)
        wav = torch.from_numpy(mixed)
        print(f"[lowvram] multiband ensemble @ {cutoff - 1000} Hz (fork v0.3)")

    return wav.unsqueeze(0)


def process_and_save(
    ld, in_path, out_dir, name, ddim_steps=50, seed=42, guidance_scale=3.5,
    chunking="auto", chunk_duration_s=10, overlap_duration_s=2, stereo_mode="split",
    out_sr=48000, out_format="wav", input_cutoff=0, multiband_ensemble=False,
    loudness_match=False, synthetic_air=0.0,
):
    """Full pipeline: optional stereo split -> per-channel AudioSR -> merge back.

    stereo_mode="split" keeps true stereo (L/R processed separately, ~2x time);
    "mono" keeps the legacy behaviour (left channel / downmix).
    The model natively restores to 48 kHz; out_sr != 48000 is a high-quality
    sinc resample of that result (e.g. 44100 for CD, 16000 for speech models).
    out_format != "wav" converts the result with ffmpeg (mp3/flac/m4a/ogg);
    the wav master is always kept next to it.

    The last three arguments are the AudioSR-Colab-Fork DSP options (all off by
    default, i.e. the stock pipeline): `input_cutoff` band-limits the model
    input (Hz, 0 = off), `multiband_ensemble` re-uses the original low band,
    `loudness_match` matches each chunk to the input LUFS instead of its peak.
    `synthetic_air` (0 = off) mixes synthesised harmonics above 24 kHz into a
    >48 kHz export - generated content, not recovered detail.
    Returns the final output path.
    """
    import shutil as _shutil
    import tempfile

    import soundfile as sf

    out_sr = int(max(8000, min(192000, int(out_sr))))
    os.makedirs(out_dir, exist_ok=True)

    split_stereo = stereo_mode == "split" and is_stereo(in_path)
    _progress_start(
        (2 if split_stereo else 1)
        * _planned_chunks(
            in_path, chunking, chunk_duration_s, overlap_duration_s, bool(loudness_match)
        )
    )

    if split_stereo:
        data_in, _sr_in = sf.read(in_path, dtype="float32", always_2d=True)
        in_peaks = [float(np.max(np.abs(data_in[:, i]))) + 1e-8 for i in range(2)]
        tmp = tempfile.mkdtemp(prefix="audiosr_st_")
        try:
            ch_paths = _split_stereo(in_path, tmp)
            outs = []
            for ci, ch in enumerate(ch_paths):
                print(f"[lowvram] processing channel: {os.path.basename(ch)}")
                w = _process_file(
                    ld, ch, ddim_steps, seed, guidance_scale,
                    chunk_duration_s, overlap_duration_s, chunking,
                    input_cutoff=input_cutoff,
                    multiband_ensemble=multiband_ensemble,
                    loudness_match=loudness_match,
                    label=f"声道 {ci + 1}/{len(ch_paths)}" if len(ch_paths) > 1 else "",
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
            stereo = _resample(stereo.T, 48000, out_sr).T
        if synthetic_air > 0:
            if out_sr > 48000:
                stereo = np.stack(
                    [
                        _synthetic_air(stereo[:, ch], out_sr, synthetic_air)
                        for ch in range(stereo.shape[1])
                    ],
                    axis=1,
                )
                print(
                    f"[lowvram] synthetic top band added (>24 kHz, amount "
                    f"{synthetic_air:g}) - generated content"
                )
            else:
                print("[lowvram] synthetic_air ignored: --out_sr must be > 48000")
        out_path = os.path.join(out_dir, name + ".wav")
        sf.write(out_path, stereo, out_sr, subtype="PCM_16")
        print(f"[lowvram] stereo merged -> {out_path} ({out_sr} Hz)")
        return convert_output(out_path, out_format)

    wav = _process_file(
        ld, in_path, ddim_steps, seed, guidance_scale,
        chunk_duration_s, overlap_duration_s, chunking,
        input_cutoff=input_cutoff,
        multiband_ensemble=multiband_ensemble,
        loudness_match=loudness_match,
    )
    if out_sr != 48000:
        wav = torch.from_numpy(
            _resample(wav.detach().cpu().float().numpy(), 48000, out_sr)
        )
    if synthetic_air > 0:
        if out_sr > 48000:
            flat = wav.detach().cpu().float().numpy().reshape(-1)
            wav = torch.from_numpy(_synthetic_air(flat, out_sr, synthetic_air)).unsqueeze(0)
            print(
                f"[lowvram] synthetic top band added (>24 kHz, amount "
                f"{synthetic_air:g}) - generated content"
            )
        else:
            print("[lowvram] synthetic_air ignored: --out_sr must be > 48000")
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
    ap.add_argument(
        "--synthetic_air",
        choices=list(AIR_LEVELS),
        default="off",
        help="add SYNTHESISED harmonics above 24 kHz so a 96/192 kHz export is not "
        "empty up top (generated content, not recovered detail); only applies when "
        "--out_sr > 48000",
    )
    ap.add_argument("--chunking", action="store_true", help="chunk long audio")
    ap.add_argument("--chunk_duration", type=int, default=10)
    ap.add_argument("--overlap_duration", type=int, default=2)
    ap.add_argument(
        "--input_cutoff",
        type=int,
        default=0,
        help="band-limit the model input to this frequency in Hz by resampling "
        "it to input_cutoff*2 Hz instead of low-pass filtering it (AudioSR-Colab-Fork "
        "v0.5); 0 = off, try 14000",
    )
    ap.add_argument(
        "--multiband_ensemble",
        action="store_true",
        help="keep the original audio below (input_cutoff - 1000) Hz and mix in the "
        "generated audio above it (AudioSR-Colab-Fork v0.3); implies --input_cutoff 14000",
    )
    ap.add_argument(
        "--loudness_match",
        action="store_true",
        help="match every processed chunk to the integrated loudness (LUFS) of its "
        "input chunk instead of its peak, removing chunk-to-chunk volume jumps "
        "(AudioSR-Colab-Fork v0.5); needs pyloudnorm",
    )
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

    if args.multiband_ensemble or args.input_cutoff:
        cutoff = args.input_cutoff or FORK_DEFAULT_CUTOFF
        print(
            f"[lowvram] fork DSP on: input_cutoff={cutoff} Hz"
            + (f", multiband ensemble @ {cutoff - 1000} Hz" if args.multiband_ensemble else "")
            + (", LUFS loudness matching" if args.loudness_match else "")
        )

    if AIR_LEVELS[args.synthetic_air] > 0:
        if args.out_sr > 48000:
            print(
                f"[lowvram] synthetic top band: {args.synthetic_air} "
                f"({AIR_LEVELS[args.synthetic_air]:g}) - synthesised harmonics above "
                "24 kHz, not recovered detail"
            )
        else:
            print(f"[lowvram] --synthetic_air {args.synthetic_air} ignored: need --out_sr > 48000")

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
        input_cutoff=args.input_cutoff,
        multiband_ensemble=args.multiband_ensemble,
        loudness_match=args.loudness_match,
        synthetic_air=AIR_LEVELS[args.synthetic_air],
    )
    print(f"[lowvram] total processing time: {time.time() - t0:.1f}s")
    print(f"[lowvram] saved to: {out}")


if __name__ == "__main__":
    main()
