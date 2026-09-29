# -*- coding: utf-8 -*-
"""Verify out_format conversion (mp3/flac/m4a/ogg) keeps a wav master and
produces valid files per ffprobe."""
import json
import os
import subprocess

import run_lowvram as lv

ld = lv.build_lowvram_model("basic", "cuda:0", lv.default_ckpt_path(), gpu_vae=True)

for fmt in ["mp3", "flac", "m4a", "ogg"]:
    out = lv.process_and_save(
        ld, r"example\sound_effect.wav", "output_test_fmt", f"mono_{fmt}",
        ddim_steps=50, seed=42, guidance_scale=3.5, chunking="auto",
        stereo_mode="mono", out_sr=48000, out_format=fmt,
    )
    wav = os.path.splitext(out)[0] + ".wav"
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-show_entries",
         "format=duration,bit_rate", "-of", "json", out],
        capture_output=True, text=True,
    )
    info = json.loads(r.stdout)
    st = info.get("streams", [{}])[0]
    fm = info.get("format", {})
    print(
        fmt,
        "->", os.path.basename(out),
        "| codec", st.get("codec_name"),
        "| sr", st.get("sample_rate"),
        "| ch", st.get("channels"),
        "| dur", round(float(fm.get("duration", 0)), 2),
        "| wav kept:", os.path.exists(wav),
        "| sizeMB", round(os.path.getsize(out) / 1048576, 2),
    )
