# -*- coding: utf-8 -*-
"""
Double-click helper: pick an audio file with a native dialog, process it with
the AudioSR low-VRAM pipeline using default settings, then open the output
folder in Explorer.

Run via  \u53cc\u51fb\u9009\u62e9\u97f3\u9891\u5904\u7406.cmd  (or `python pick_and_run.py`).
"""
import os

import torchaudio

import run_lowvram as lowvram
from audiosr import get_time

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = lowvram.default_ckpt_path()


def ensure_decodable(path):
    return lowvram.ensure_decodable(path)


def pick_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择要处理的音频（5~10 秒片段效果最佳）",
        filetypes=[
            ("音频文件", "*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.wma"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return path


def main():
    path = pick_file()
    if not path:
        print("未选择文件，已退出。")
        return
    src = ensure_decodable(path)
    try:
        info = torchaudio.info(src)
        duration = info.num_frames / info.sample_rate
    except Exception as e:
        print(f"无法读取该音频：{e}")
        return
    print(f"输入：{path}")
    print(f"时长：{duration:.2f} 秒")

    ld = lowvram.build_lowvram_model("basic", "cuda:0", CKPT, gpu_vae=True)
    if lowvram.is_stereo(src):
        print("检测到双声道输入：将逐声道修复（真立体声，处理时间约 2 倍）")
    else:
        print("单声道输入。")
    out_dir = os.path.join(BASE, "output", get_time())
    name = os.path.splitext(os.path.basename(src))[0] + "_AudioSR_48K"
    out = lowvram.process_and_save(
        ld, src, out_dir, name,
        ddim_steps=50, seed=42, guidance_scale=3.5,
        chunking="auto", chunk_duration_s=10, overlap_duration_s=2,
        stereo_mode="split",
    )
    print(f"输出文件：{out}")
    try:
        os.startfile(os.path.dirname(out))
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    input("按回车退出...")
