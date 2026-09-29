# -*- coding: utf-8 -*-
"""Verify out_sr (output sample-rate resampling) for both mono and stereo."""
import os

import numpy as np
import soundfile as sf

import run_lowvram as lv

ld = lv.build_lowvram_model("basic", "cuda:0", lv.default_ckpt_path(), gpu_vae=True)

out_mono = lv.process_and_save(
    ld, r"example\sound_effect.wav", "output_test_sr", "mono_44100",
    ddim_steps=50, seed=42, guidance_scale=3.5, chunking="auto",
    stereo_mode="mono", out_sr=44100,
)
print("mono ->", out_mono)

out_st = lv.process_and_save(
    ld, r"test_stereo.wav", "output_test_sr", "stereo_44100",
    ddim_steps=50, seed=42, guidance_scale=3.5, chunking="auto",
    stereo_mode="split", out_sr=44100,
)
print("stereo ->", out_st)

for p in (out_mono, out_st):
    info = sf.info(p)
    x, _ = sf.read(p, dtype="float32", always_2d=True)
    pL = float(np.max(np.abs(x[:, 0])))
    ratio = round(float(np.max(np.abs(x[:, 1])) / pL), 3) if x.shape[1] > 1 else None
    print(
        os.path.basename(p),
        "| ch", info.channels,
        "| sr", info.samplerate,
        "| dur", round(info.duration, 2),
        "| ratioL/R", ratio,
        "| NaN", bool(np.isnan(x).any()),
    )
