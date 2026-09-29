# -*- coding: utf-8 -*-
"""
AudioSR local Web UI (4 GB VRAM friendly).

Run:
    .venv\\Scripts\\python.exe webui_server.py        (or double-click the .cmd)

Then open http://127.0.0.1:7860 in a browser (done automatically).

LAN mode (phones / other PCs on the same Wi-Fi):
    .venv\\Scripts\\python.exe webui_server.py --host 0.0.0.0
  The console prints the reachable http://<this-pc-ip>:<port> URL(s).

The model is loaded ONCE and kept in memory; every job reuses it.
Uploads are converted to mono float wav with ffmpeg when needed,
then processed through the low-VRAM pipeline from `run_lowvram.py`.
"""
import argparse
import cgi
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import run_lowvram as lowvram  # sets NO_PROXY/HF_ENDPOINT and patches attention

import numpy as np  # noqa: E402
import torch  # noqa: E402

from audiosr import save_wave, get_time  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "webui_uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CKPT = lowvram.default_ckpt_path()

DECODABLE_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}

_ld = None
_ld_lock = threading.Lock()
_ld_ready = False
_model_error = None
_gpu_vae_mode = None

_jobs = {}
_jobs_lock = threading.Lock()
_queue = queue.Queue()


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------
def ensure_decodable(path):
    """Convert exotic containers (m4a/aac/wma/...) to mono wav via ffmpeg."""
    return lowvram.ensure_decodable(path)


def get_duration(path):
    try:
        import torchaudio

        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        return None


# ---------------------------------------------------------------------------
# model management
# ---------------------------------------------------------------------------
def ckpt_present():
    """True when a local checkpoint can be used (no download needed)."""
    try:
        return os.path.exists(lowvram.default_ckpt_path())
    except Exception:
        return False


def get_model():
    global _ld, _ld_ready, _model_error
    with _ld_lock:
        if _ld is None:
            try:
                _ld = lowvram.build_lowvram_model("basic", "cuda:0", CKPT)
                _ld_ready = True
                _model_error = None
            except BaseException as exc:  # surface it in the UI instead of spinning
                _model_error = f"{type(exc).__name__}: {exc}"
                raise
        return _ld


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
class Tee(object):
    """Capture stdout/stderr of the processing code into the job log."""

    def __init__(self, job):
        self.job = job
        self._orig_out = None
        self._orig_err = None
        self._buf = ""

    def __enter__(self):
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *a):
        sys.stdout, sys.stderr = self._orig_out, self._orig_err

    def write(self, s):
        try:
            self._orig_out.write(s)
        except Exception:
            pass
        self._buf += s
        while True:
            hits = [i for i in (self._buf.find("\n"), self._buf.find("\r")) if i != -1]
            if not hits:
                break
            i = min(hits)
            line = self._buf[:i].strip()
            self._buf = self._buf[i + 1:]
            if line:
                self.job.add_log(line)

    def flush(self):
        try:
            self._orig_out.flush()
        except Exception:
            pass


class Job(object):
    def __init__(self, job_id, in_path, display_name, params, duration, stereo_active=False):
        self.id = job_id
        self.in_path = in_path
        self.display_name = display_name
        self.params = params
        self.duration = duration
        self.stereo_active = stereo_active
        self.state = "queued"  # queued | running | done | error
        self.log = []
        self.error = None
        self.result = None
        self.t_created = time.time()
        self.t_start = None
        self.t_end = None
        self.chunk_start = None
        self.chunk_end = None
        self.ddim_pct = None

    def add_log(self, line):
        self.log.append(line)
        del self.log[:-400]
        m = re.search(r"from ([\d.]+)s to ([\d.]+)s", line)
        if m:
            self.chunk_start, self.chunk_end = float(m.group(1)), float(m.group(2))
            self.ddim_pct = None
            return
        if "DDIM Sampler:" in line:
            m = re.search(r"(\d+)%", line)
            if m:
                self.ddim_pct = int(m.group(1))

    def percent(self):
        if self.state == "done":
            return 100.0
        dur = self.duration or 0
        if duration_unknown := (dur <= 0):
            return None if self.ddim_pct is None else float(min(99, self.ddim_pct))
        try:
            if self.chunk_start is not None:
                base = self.chunk_start / dur * 100.0
                span = max(self.chunk_end - self.chunk_start, 0.0) / dur * 100.0
                inner = (self.ddim_pct or 0) / 100.0
                return float(min(99.0, max(1.0, base + span * inner)))
            if self.ddim_pct is not None:
                return float(min(99.0, max(1.0, self.ddim_pct * 0.8)))
        except Exception:
            pass
        return 3.0

    def to_json(self):
        elapsed = (self.t_end or time.time()) - (self.t_start or self.t_created)
        return {
            "id": self.id,
            "state": self.state,
            "message": (self.log[-1] if self.log else "等待开始…"),
            "log": self.log[-12:],
            "elapsed": round(elapsed, 1),
            "queue_pos": _queue.qsize() if self.state == "queued" else 0,
            "percent": self.percent(),
            "error": self.error,
            "result_url": f"/api/result?id={self.id}" if self.result else None,
            "result_path": self.result,
            "result_name": os.path.basename(self.result) if self.result else None,
            "stereo_active": self.stereo_active,
        }


def run_job(job):
    global _gpu_vae_mode
    ld = get_model()
    p = job.params
    want_gpu_vae = bool(p.get("gpu_vae"))
    if _gpu_vae_mode != want_gpu_vae:
        lowvram.set_gpu_vae(ld, want_gpu_vae)
        _gpu_vae_mode = want_gpu_vae
    src = ensure_decodable(job.in_path)
    with Tee(job):
        if p["stereo"] == "split" and lowvram.is_stereo(src):
            print("[webui] stereo input -> per-channel processing (true stereo, ~2x time)")
        out_dir = os.path.join(OUTPUT_DIR, get_time())
        sr_tag = f"{p['out_sr'] / 1000:g}k"
        name = os.path.splitext(job.display_name)[0] + f"_AudioSR_{sr_tag}"
        job.result = lowvram.process_and_save(
            ld, src, out_dir, name,
            ddim_steps=p["steps"], seed=p["seed"], guidance_scale=p["gs"],
            chunking=p["chunking"], chunk_duration_s=p["chunk_dur"],
            overlap_duration_s=2, stereo_mode=p["stereo"], out_sr=p["out_sr"],
            out_format=p["out_format"],
        )
        print(f"[webui] done -> {job.result}")


def worker_loop():
    while True:
        job = _queue.get()
        job.state = "running"
        job.t_start = time.time()
        try:
            run_job(job)
            job.state = "done"
        except Exception as e:
            job.state = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.add_log("ERROR: " + traceback.format_exc())
        finally:
            job.t_end = time.time()
            _queue.task_done()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "AudioSRWebUI/1.0"

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _find_job(self):
        q = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(q.query)
        jid = (params.get("id") or [""])[0]
        with _jobs_lock:
            return _jobs.get(jid)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/health":
            self._json({
                "model_ready": _ld_ready,
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "busy": any(j.state == "running" for j in _jobs.values()),
                "queued": _queue.qsize(),
                "local_ckpt": ckpt_present(),
                "model_error": _model_error,
            })
        elif path == "/api/status":
            job = self._find_job()
            if not job:
                self._json({"error": "job not found"}, 404)
            else:
                self._json(job.to_json())
        elif path == "/api/result":
            job = self._find_job()
            if not job or not job.result or not os.path.isfile(job.result):
                self._json({"error": "result not ready"}, 404)
                return
            with open(job.result, "rb") as f:
                data = f.read()
            fname = os.path.basename(job.result)
            ext = os.path.splitext(job.result)[1].lower()
            mime = {
                ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
                ".m4a": "audio/mp4", ".ogg": "audio/ogg",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Disposition",
                "inline; filename=\"output.wav\"; "
                f"filename*=UTF-8''{urllib.parse.quote(fname)}",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/process":
            self._handle_process()
        elif path == "/api/open":
            job = self._find_job()
            if job and job.result:
                try:
                    os.startfile(os.path.dirname(job.result))  # noqa: only on Windows
                    self._json({"ok": True})
                except Exception as e:
                    self._json({"error": str(e)}, 500)
            else:
                self._json({"error": "no result"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def _handle_process(self):
        try:
            env = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            }
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=env)
            if "audio" not in form or not getattr(form["audio"], "filename", ""):
                self._json({"error": "缺少音频文件"}, 400)
                return
            item = form["audio"]
            orig_name = os.path.basename(item.filename) or "upload.wav"
            safe_ext = os.path.splitext(orig_name)[1][:10]
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            up_path = os.path.join(
                UPLOAD_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{safe_ext}"
            )
            with open(up_path, "wb") as f:
                shutil.copyfileobj(item.file, f)

            def _num(key, default, cast):
                try:
                    return cast(form.getvalue(key, str(default)))
                except Exception:
                    return default

            chunk_raw = str(form.getvalue("chunking", "auto")).lower()
            if chunk_raw in ("1", "true", "yes"):
                chunking = True
            elif chunk_raw in ("0", "false", "no"):
                chunking = False
            else:
                chunking = None  # auto: decide from duration
            stereo_raw = str(form.getvalue("stereo", "split")).lower()
            gpu_vae_raw = str(form.getvalue("gpu_vae", "0")).lower()
            out_fmt_raw = str(form.getvalue("out_format", "wav")).lower()
            params = {
                "steps": max(5, min(200, _num("steps", 50, int))),
                "seed": _num("seed", 42, int),
                "gs": max(1.0, min(10.0, _num("gs", 3.5, float))),
                "chunk_dur": max(5, min(30, _num("chunk_dur", 10, int))),
                "chunking": chunking,
                "stereo": stereo_raw if stereo_raw in ("split", "mono") else "split",
                "gpu_vae": gpu_vae_raw in ("1", "true", "yes", "on"),
                "out_sr": max(8000, min(192000, _num("out_sr", 48000, int))),
                "out_format": out_fmt_raw if out_fmt_raw in ("wav", "mp3", "flac", "m4a", "ogg") else "wav",
            }
            probe = ensure_decodable(up_path)
            duration = get_duration(probe)
            stereo_active = lowvram.is_stereo(probe) and params["stereo"] == "split"
            job = Job(uuid.uuid4().hex[:12], up_path, orig_name, params, duration, stereo_active)
            with _jobs_lock:
                _jobs[job.id] = job
                if len(_jobs) > 60:  # prune old finished jobs
                    for k in sorted(_jobs, key=lambda k: _jobs[k].t_created)[:20]:
                        if _jobs[k].state in ("done", "error"):
                            del _jobs[k]
            _queue.put(job)
            self._json({
                "id": job.id,
                "duration": duration,
                "stereo_active": stereo_active,
                "params": params,
            })
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):  # keep the console tidy
        pass


# ---------------------------------------------------------------------------
# UI page
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AudioSR 音频超分辨率 · 本地版</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#2d333b;--fg:#e6edf3;--dim:#8b949e;--acc:#2563eb;--acc2:#3b82f6;--ok:#16a34a;--warn:#d97706;--err:#dc2626}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 20% -10%,#1e293b 0%,var(--bg) 60%);color:var(--fg);font:15px/1.6 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 80px}
header h1{margin:0;font-size:26px;background:linear-gradient(90deg,#60a5fa,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent}
header p{margin:4px 0 18px;color:var(--dim);font-size:14px}
.health{display:inline-block;margin-bottom:14px;padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--card);font-size:13px;color:var(--dim)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:14px}
#dropzone{border:2px dashed #3b4048;border-radius:12px;padding:34px 16px;text-align:center;color:var(--dim);transition:.15s;cursor:pointer}
#dropzone.hover{border-color:var(--acc2);background:#1c2533;color:var(--fg)}
#dropzone b{color:var(--fg)}
.btn{display:inline-block;border:0;border-radius:10px;padding:10px 18px;font-size:15px;cursor:pointer;text-decoration:none;color:#fff;background:#30363d}
.btn:hover{filter:brightness(1.15)}.btn.primary{background:linear-gradient(135deg,var(--acc),var(--acc2))}.btn.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
.btn[disabled]{opacity:.45;cursor:not-allowed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
label{font-size:13px;color:var(--dim);display:block}
input[type=number],select{width:100%;margin-top:5px;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:#0d1117;color:var(--fg)}
.row{display:flex;gap:8px;align-items:center}.row>*{flex:0 0 auto}
.mini{padding:6px 10px;font-size:12px;border-radius:8px;border:1px solid var(--line);background:#0d1117;color:var(--fg);cursor:pointer}
.hint{color:var(--dim);font-size:13px;margin:12px 0 0}
.badge{display:inline-block;padding:4px 12px;border-radius:999px;font-size:13px;border:1px solid var(--line)}
.badge.running{color:#93c5fd;border-color:#1d4ed8;background:#172554}
.badge.done{color:#86efac;border-color:#15803d;background:#052e16}
.badge.error{color:#fca5a5;border-color:#b91c1c;background:#450a0a}
.bar{height:10px;border-radius:999px;background:#0d1117;border:1px solid var(--line);overflow:hidden;margin:12px 0}
#barfill{height:100%;width:0%;background:linear-gradient(90deg,var(--acc),#a78bfa);transition:width .4s}
pre#log{background:#0d1117;border:1px solid var(--line);border-radius:10px;padding:10px;font-size:12px;color:#9aa4b2;max-height:150px;overflow:auto;white-space:pre-wrap;margin:8px 0 0}
audio{width:100%;margin:10px 0}
#fileinfo{font-size:13px;color:var(--dim);margin-top:10px}
.help{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}
.help .item{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#0d1117}
.help .item b{display:block;color:#93c5fd;font-size:13px;margin-bottom:4px}
.help .item span{font-size:12.5px;color:var(--dim);line-height:1.6;display:block}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>AudioSR · 音频超分辨率</h1>
    <p>任意音频 → 48 kHz 高保真 ｜ 4GB 显存优化版 ｜ 本地运行，不上传云端</p>
  </header>
  <div class="health" id="health">正在连接…</div>

  <section class="card">
    <div id="dropzone">
      <div style="font-size:34px">🎵</div>
      <div>拖拽音频文件到这里，或 <b>点击选择文件</b></div>
      <div style="font-size:12px;margin-top:6px">支持 wav / flac / mp3 / ogg；m4a、wma 等会自动用 ffmpeg 转换</div>
      <input type="file" id="file" accept="audio/*,.wav,.flac,.mp3,.ogg,.m4a,.aac,.wma" hidden>
    </div>
    <div id="fileinfo"></div>
  </section>

  <section class="card">
    <div class="grid">
      <label>采样步数 DDIM<input type="number" id="steps" value="50" min="10" max="200"></label>
      <label>随机种子
        <span class="row"><input type="number" id="seed" value="42" style="flex:1"><button class="mini" id="rnd">随机</button></span>
      </label>
      <label>引导强度 -gs<input type="number" id="gs" value="3.5" step="0.5" min="1" max="10"></label>
      <label>分块时长（秒）<input type="number" id="chunkdur" value="10" min="5" max="30"></label>
      <label>输出采样率
        <select id="outsr">
          <option value="48000" selected>48 kHz（原生）</option>
          <option value="44100">44.1 kHz（CD）</option>
          <option value="96000">96 kHz（仅容器升采样）</option>
          <option value="32000">32 kHz</option>
          <option value="24000">24 kHz</option>
          <option value="16000">16 kHz（语音）</option>
          <option value="192000">192 kHz（仅容器升采样）</option>
        </select>
      </label>
      <label>输出格式
        <select id="outfmt">
          <option value="wav" selected>WAV（无损，原生）</option>
          <option value="flac">FLAC（无损压缩）</option>
          <option value="mp3">MP3 320k</option>
          <option value="m4a">M4A / AAC 256k</option>
          <option value="ogg">OGG Vorbis</option>
        </select>
      </label>
      <label>双声道处理
        <select id="stereo">
          <option value="split" selected>逐声道修复（真立体声）</option>
          <option value="mono">混合单声道（更快）</option>
        </select>
      </label>
    </div>
    <p class="hint">超过 12 秒的音频会自动分块（块间 2 秒交叉淡化）。勾选 GPU 加速时，单声道约 2–3 倍音频时长、立体声约 4–6 倍；5–10 秒的片段效果最佳。</p>
    <label style="display:flex;gap:8px;align-items:center;margin-top:12px;color:var(--fg);font-size:14px;cursor:pointer">
      <input type="checkbox" id="gpuvae" checked> VAE・声码器也搬到 GPU（fp16）— 大幅提速，占用更多显存
    </label>
    <div style="margin-top:14px"><button class="btn primary" id="go" disabled>开始处理</button></div>
  </section>

  <section class="card" id="progresscard" hidden>
    <div class="row" style="justify-content:space-between"><span class="badge" id="state">排队中</span><span style="color:var(--dim);font-size:13px" id="elapsed"></span></div>
    <div class="bar"><div id="barfill"></div></div>
    <pre id="log"></pre>
  </section>

  <section class="card" id="resultcard" hidden>
    <h2 style="margin:0 0 6px;font-size:18px">处理完成 🎉</h2>
    <audio id="player" controls preload="metadata"></audio>
    <div class="row" style="flex-wrap:wrap">
      <a class="btn primary" id="dl" download>下载 WAV</a>
      <button class="btn ghost" id="openfolder">打开输出文件夹</button>
      <button class="btn ghost" id="again">处理其他文件</button>
    </div>
    <p class="hint" id="resultpath"></p>
  </section>

  <section class="card" id="helpcard">
    <h2 style="margin:0 0 12px;font-size:16px">📖 选项说明</h2>
    <div class="help">
      <div class="item"><b>采样步数（DDIM）</b><span>去噪迭代次数，默认 50（官方推荐）。调高（80–100）细节略更精细但更慢；调低（20–30）更快，质感略糙。</span></div>
      <div class="item"><b>随机种子</b><span>高频重建的随机起点。同一段音频换种子结果会不同——觉得某处高频不自然时，换个数字（或点「随机」）重跑即可。</span></div>
      <div class="item"><b>引导强度（-gs）</b><span>默认 3.5。越大越贴近原始低频、越保守；越小越“自由发挥”，可能出现更多高频但失真风险也高。</span></div>
      <div class="item"><b>分块时长</b><span>超过 12 秒的音频自动分块处理（块间 2 秒交叉淡化）。默认 10 秒/块；一般不用改。</span></div>
      <div class="item"><b>输出采样率</b><span>模型原生 48 kHz（推荐）。44.1 kHz 适合 CD/音乐库；96k/192k 仅把容器升采样，24 kHz 以上没有新内容；16k 适合语音识别输入。</span></div>
      <div class="item"><b>输出格式</b><span>WAV 无损原生（推荐归档）。MP3 320k / M4A·AAC 256k / OGG 体积小适合分享；FLAC 无损压缩；转换由 ffmpeg 自动完成，WAV 母版会同时保留。</span></div>
      <div class="item"><b>双声道处理</b><span>「逐声道修复」：左右分别修复再合并，保留真立体声，耗时约 ×2，推荐给歌曲；「混合单声道」更快，但立体声信息会被合并。</span></div>
      <div class="item"><b>GPU 加速（VAE）</b><span>把声码器和两个 VAE 也搬到显卡（fp16），速度约快 3 倍；峰值显存约 2.9 GB。如果同时玩游戏/看视频，取消勾选可省显存（速度约慢 3 倍）。</span></div>
      <div class="item"><b>耗时参考</b><span>GPU 加速下：单声道约音频时长的 2–3 倍；立体声（逐声道）约 4–6 倍。长音频请耐心等待进度条。</span></div>
    </div>
    <p class="hint" style="margin-top:12px">支持格式：wav / flac / mp3 / ogg（m4a、wma 等自动经 ffmpeg 转换）；输出保存到程序目录的 output 文件夹，并可在页面直接播放/下载；全程本地运行，不上传任何数据。5–10 秒片段效果最佳。</p>
  </section>
</div>

<script>
const $ = s => document.querySelector(s);
let selectedFile = null, currentId = null, modelReady = false;
let fileDuration = null;

function fmt(sec){ if(sec==null) return "—"; sec=Math.round(sec); const m=Math.floor(sec/60), s=sec%60; return m? `${m}分${s}秒` : `${s}秒`; }

async function health(){
  try{
    const j = await (await fetch("/api/health")).json();
    modelReady = j.model_ready;
    let st, color = "";
    if (j.model_error){
      st = "✗ 模型加载失败：" + j.model_error;
      color = "#f87171";
    } else if (modelReady){
      st = `● 模型已就绪 · ${j.device}` + (j.busy ? " · 正在处理任务" : (j.queued ? ` · 队列 ${j.queued}` : " · 空闲"));
    } else if (j.local_ckpt){
      st = "◌ 模型加载中（首次约 1 分钟，硬盘较慢时更久，请稍候）…";
    } else {
      st = "◌ 首次运行：正在下载模型权重（约 5.75 GB，进度见程序窗口）…";
    }
    $("#health").textContent = st;
    $("#health").style.color = color;
  }catch(e){ $("#health").textContent = "✗ 无法连接本地服务"; }
  $("#go").disabled = !modelReady || !selectedFile || currentId !== null;
}
setInterval(health, 2000); health();

function setFile(f){
  selectedFile = f; fileDuration = null;
  if(!f){ $("#fileinfo").textContent=""; return; }
  $("#fileinfo").textContent = `已选择：${f.name}（${(f.size/1048576).toFixed(1)} MB）`;
  const a = document.createElement("audio"); a.preload = "metadata";
  a.onloadedmetadata = () => { fileDuration = a.duration; $("#fileinfo").textContent = `已选择：${f.name}（${(f.size/1048576).toFixed(1)} MB · 时长 ${fmt(a.duration)}）`; };
  a.src = URL.createObjectURL(f);
  health();
}
$("#dropzone").onclick = () => $("#file").click();
$("#file").onchange = e => setFile(e.target.files[0]);
["dragover","dragenter"].forEach(ev => $("#dropzone").addEventListener(ev, e => { e.preventDefault(); $("#dropzone").classList.add("hover"); }));
["dragleave","drop"].forEach(ev => $("#dropzone").addEventListener(ev, e => { e.preventDefault(); $("#dropzone").classList.remove("hover"); }));
$("#dropzone").addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if(f) setFile(f); });
$("#rnd").onclick = () => { $("#seed").value = Math.floor(Math.random()*100000); };

$("#go").onclick = async () => {
  if(!selectedFile) return;
  const fd = new FormData();
  fd.append("audio", selectedFile);
  fd.append("steps", $("#steps").value);
  fd.append("seed", $("#seed").value);
  fd.append("gs", $("#gs").value);
  fd.append("chunk_dur", $("#chunkdur").value);
  fd.append("stereo", $("#stereo").value);
  fd.append("gpu_vae", $("#gpuvae").checked ? "1" : "0");
  fd.append("out_sr", $("#outsr").value);
  fd.append("out_format", $("#outfmt").value);
  $("#resultcard").hidden = true; $("#progresscard").hidden = false;
  $("#state").className = "badge"; $("#state").textContent = "上传中…";
  try{
    const r = await fetch("/api/process", {method:"POST", body: fd});
    const j = await r.json();
    if(j.error){ $("#state").className="badge error"; $("#state").textContent="失败"; $("#log").textContent = j.error; currentId=null; health(); return; }
    currentId = j.id; poll();
  }catch(e){ $("#state").className="badge error"; $("#state").textContent="网络错误"; $("#log").textContent = String(e); currentId = null; health(); }
};

async function poll(){
  if(!currentId) return;
  try{
    const j = await (await fetch("/api/status?id=" + currentId)).json();
    const st = {queued:"排队中",running:"处理中",done:"完成",error:"失败"}[j.state] || j.state;
    $("#state").textContent = st + (j.queue_pos ? `（前面还有 ${j.queue_pos} 个任务）` : "");
    $("#state").className = "badge " + j.state;
    const speedFactor = ($("#gpuvae").checked ? 2.5 : 5.5) * (j.stereo_active ? 2 : 1);
    $("#elapsed").textContent = `已用时 ${fmt(j.elapsed)}` + (fileDuration ? ` · 预计共约 ${fmt(fileDuration*speedFactor)}` : "");
    if(j.percent != null) $("#barfill").style.width = j.percent + "%";
    $("#log").textContent = (j.log||[]).join("\n");
    $("#log").scrollTop = $("#log").scrollHeight;
    if(j.state === "done"){
      const url = j.result_url + "&t=" + Date.now();
      $("#player").src = url; $("#dl").href = url; $("#dl").setAttribute("download", j.result_name || "output.wav");
      const ext = ((j.result_name || "output.wav").split(".").pop() || "wav").toUpperCase();
      $("#dl").textContent = "下载 " + ext;
      let text = "已保存到：" + j.result_path;
      if(ext !== "WAV") text += " ｜ 无损 WAV 母版已同目录保留";
      $("#resultpath").textContent = text;
      $("#resultcard").hidden = false; currentId = null; health(); return;
    }
    if(j.state === "error"){ currentId = null; health(); return; }
  }catch(e){ /* keep polling */ }
  setTimeout(poll, 1000);
}

$("#openfolder").onclick = () => fetch("/api/open?id=" + (currentId || (new URL($("#player").src, location.href)).searchParams.get("id")), {method:"POST"});
$("#again").onclick = () => { $("#resultcard").hidden = true; $("#progresscard").hidden = true; setFile(null); health(); };
</script>
</body>
</html>
"""


def lan_ips():
    """Best-effort list of this machine's non-loopback IPv4 addresses."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # no packets sent, just picks the route
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                ips.append(ip)
        finally:
            s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=get_model, daemon=True).start()  # preload

    port = args.port
    for attempt in range(5):
        try:
            httpd = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError:
            print(f"[webui] port {port} is busy, trying {port + 1}")
            port += 1
    else:
        print("[webui] no free port found")
        sys.exit(1)

    if args.host in ("0.0.0.0", "::", ""):
        # LAN mode: open the loopback URL locally, advertise the LAN URLs too
        url = f"http://127.0.0.1:{port}"
        print(f"[webui] serving at {url}  (Ctrl+C to stop)")
        ips = lan_ips()
        if ips:
            print("[webui] LAN access from phones / other PCs on the same network:")
            for ip in ips:
                print(f"[webui]   http://{ip}:{port}")
        else:
            print("[webui] LAN mode: use this PC's IPv4 address with the port above")
        print("[webui] if remote devices cannot connect, allow Python through the")
        print("[webui] Windows Firewall (private network) when prompted")
    else:
        url = f"http://{args.host}:{port}"
        print(f"[webui] serving at {url}  (Ctrl+C to stop)")
    print("[webui] model will load in the background; the page shows the status")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] bye")


if __name__ == "__main__":
    main()
