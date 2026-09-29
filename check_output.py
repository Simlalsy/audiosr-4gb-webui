# -*- coding: utf-8 -*-
"""Quick sanity check for the AudioSR low-VRAM run.

Usage:
    python check_output.py [input.wav] [output.wav]

Compares the high-band energy of the input and the generated output, then
prints the checkpoint parameter distribution. Without an output argument the
newest ``*_AudioSR_*.wav`` below ``output/`` is used.
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))


def default_ckpt():
    env = os.environ.get("AUDIOSR_CKPT")
    if env:
        return env
    return os.path.join(os.path.dirname(BASE), "pytorch_model.bin")


def latest_output():
    files = glob.glob(os.path.join(BASE, "output", "*", "*_AudioSR_*.wav"))
    return max(files, key=os.path.getmtime) if files else ""


_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("input", nargs="?", default=os.path.join("example", "music.wav"))
_ap.add_argument("output", nargs="?", default="")
_ap.add_argument("--ckpt", default=default_ckpt())
_args = _ap.parse_args()

IN = _args.input
OUT = _args.output or latest_output()
CKPT = _args.ckpt
if not OUT:
    raise SystemExit(
        "no output wav found - pass one explicitly, e.g.\n"
        r"  python check_output.py example/music.wav output\<timestamp>\music_AudioSR_Processed_48K.wav"
    )


def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return x, sr


def hf_ratio(x, sr, lo=1500.0, hi=17000.0, dur=10.0):
    n = min(len(x), int(sr * dur))
    seg = x[:n] * np.hanning(n)
    S = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    total = S[f > lo].sum()
    high = S[f > hi].sum()
    return float(high / max(total, 1e-12))


a, sr_a = load(IN)
b, sr_b = load(OUT)
print(f"input : {len(a) / sr_a:.2f} s @ {sr_a} Hz")
print(f"output: {len(b) / sr_b:.2f} s @ {sr_b} Hz")
print(f"high-band energy ratio (>17 kHz), input : {hf_ratio(a, sr_a):.8f}")
print(f"high-band energy ratio (>17 kHz), output: {hf_ratio(b, sr_b):.8f}")

import torch  # noqa: E402

sd = torch.load(CKPT, map_location="cpu")["state_dict"]
tot = defaultdict(float)
for k, v in sd.items():
    if hasattr(v, "numel"):
        parts = k.split(".")
        tot[parts[0] + ("." + parts[1] if len(parts) > 2 else "")] += v.numel()
print("\ncheckpoint parameter distribution (top 15):")
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:15]:
    print(f"  {k:50s} {v / 1e6:8.1f} M   fp32 = {v * 4 / 2**30:.2f} GiB")
