# -*- coding: utf-8 -*-
"""Verify a 192 kHz export: container rate is 192k, duration intact, and the
spectrum above 24 kHz is essentially empty (proof that nothing new is invented)."""
import os

import numpy as np
import soundfile as sf

import run_lowvram as lv

ld = lv.build_lowvram_model("basic", "cuda:0", lv.default_ckpt_path(), gpu_vae=True)

out = lv.process_and_save(
    ld, r"example\sound_effect.wav", "output_test_sr", "mono_192k",
    ddim_steps=50, seed=42, guidance_scale=3.5, chunking="auto",
    stereo_mode="mono", out_sr=192000,
)

info = sf.info(out)
x, _ = sf.read(out, dtype="float32")
print(
    os.path.basename(out),
    "| ch", info.channels,
    "| sr", info.samplerate,
    "| dur", round(info.duration, 2),
    "| sizeMB", round(os.path.getsize(out) / 1048576, 2),
    "| NaN", bool(np.isnan(x).any()),
)

n = len(x)
f = np.fft.rfftfreq(n, 1 / info.samplerate)
mag = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
total = float(mag[f > 20].sum())
above24 = float(mag[f > 24000].sum())
above48 = float(mag[f > 48000].sum())
print("energy >24 kHz / total :", f"{above24 / total:.3e}")
print("energy >48 kHz / total :", f"{above48 / total:.3e}")
