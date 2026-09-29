# -*- coding: utf-8 -*-
"""打包两个发行版 ZIP：A 版（自动下载模型）/ B 版（完全离线）。

用法（在仓库根目录执行）：
    .venv\\Scripts\\python.exe release\\build_release.py [版本号] [引用] [--with-offline] [--with-model]

    版本号           默认 v1.0.0
    引用             默认 HEAD（也可写 main 或某个 tag）
    --with-offline   额外产出「B-完全离线」包（启动脚本内置 AUDIOSR_OFFLINE=1）
    --with-model     额外把模型权重切成 <2 GiB 的分卷（GitHub 单附件上限），
                     供完全离线包/离线分发使用

    版本号   默认 v1.0.0
    引用     默认 HEAD（也可写 main 或某个 tag）

产物（dist\\ 目录）：
    AudioSR-LowVRAM-<版本>-A-online-model-download.zip
    AudioSR-LowVRAM-<版本>-B-fully-offline.zip
    release_body.md   发布说明（含两个包的 SHA-256），供 `gh release create --notes-file` 使用

ZIP 内部的顶层文件夹与 使用说明.txt 仍是中文（Windows 资源管理器可正常解压），
只有对外的文件名用 ASCII —— 部分工具链会把非 ASCII 文件名丢掉。

两个包的源码完全相同（来自 `git archive`，并排除 release\\ 与 dist\\），差异只有两处：
    1. 包内根目录的 使用说明.txt 分别为 A / B 版；
    2. B 版的 4 个 .cmd 启动脚本会插入 `set AUDIOSR_OFFLINE=1`，运行时绝不联网下载模型。
"""
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_DIR = os.path.join(ROOT, "release")
DIST_DIR = os.path.join(ROOT, "dist")

EXCLUDED_PREFIXES = ("release/", "dist/", ".check_")
PART_LIMIT = 1_900_000_000  # bytes; GitHub allows < 2 GiB per release asset
MODEL_SHA256 = "8a3506b9619ed32435ce2c115604750c7bddb5ad8502be7b1a3131bde878aa01"
CMD_FILES = (
    "run_audiosr.cmd",
    "双击启动网页界面.cmd",
    "双击启动局域网网页界面.cmd",
    "双击选择音频处理.cmd",
)
ONLINE_VARIANT = {
    "tag": "A-自动下载模型",
    "slug": "A-online-model-download",
    "doc": "使用说明_自动下载模型版.txt",
    "offline": False,
}
OFFLINE_VARIANT = {
    "tag": "B-完全离线",
    "slug": "B-fully-offline",
    "doc": "使用说明_完全本地版.txt",
    "offline": True,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_model():
    """Locate the pytorch_model.bin to slice (env / app dir / models / parent)."""
    env = os.environ.get("AUDIOSR_CKPT")
    if env and os.path.exists(env):
        return env
    for rel in ("pytorch_model.bin", os.path.join("models", "pytorch_model.bin")):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            return p
    parent = os.path.join(os.path.dirname(ROOT), "pytorch_model.bin")
    if os.path.exists(parent):
        return parent
    return None


def split_model(src, out_dir):
    """Slice the checkpoint into <2 GiB parts named pytorch_model.bin.part-NN."""
    size = os.path.getsize(src)
    count = max(1, math.ceil(size / PART_LIMIT))
    chunk = math.ceil(size / count)
    results = []
    with open(src, "rb") as f:
        for i in range(count):
            name = f"pytorch_model.bin.part-{i + 1:02d}"
            dest = os.path.join(out_dir, name)
            written = 0
            with open(dest, "wb") as out:
                while written < chunk:
                    block = f.read(min(1 << 24, chunk - written))
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
            results.append((name, os.path.getsize(dest) / 2 ** 20, sha256(dest)))
            print(
                f"[ok] {name}  {os.path.getsize(dest) / 2 ** 20:.0f} MB  "
                f"sha256={results[-1][2]}"
            )
            if written == 0:
                os.remove(dest)
                results.pop()
                break
    return results


def crlf(path):
    """Normalise a text file to CRLF (cmd scripts + Notepad friendliness)."""
    data = open(path, "rb").read()
    text = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    open(path, "wb").write(text.replace(b"\n", b"\r\n"))


def patch_offline(path):
    """Insert `set AUDIOSR_OFFLINE=1` right after the HF_ENDPOINT line."""
    text = open(path, "rb").read().decode("utf-8")
    lines = text.replace("\r\n", "\n").split("\n")
    if any(ln.strip() == "set AUDIOSR_OFFLINE=1" for ln in lines):
        return
    out = []
    for line in lines:
        out.append(line)
        if line.startswith("set HF_ENDPOINT="):
            out.append("set AUDIOSR_OFFLINE=1")
    open(path, "wb").write("\r\n".join(out).encode("utf-8"))


def with_version(text, version):
    """Add a version banner line right below the document header."""
    lines = text.split("\n")
    idx = None
    for i, line in enumerate(lines[:6]):
        if i > 0 and line.startswith("===="):
            idx = i
            break
    stamp = f" 本包版本：{version}（构建于 {time.strftime('%Y-%m-%d')}）"
    if idx is None:
        return stamp + "\n" + text
    lines.insert(idx + 1, stamp)
    return "\n".join(lines)


def extract(archive, dest):
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/") or name.startswith(EXCLUDED_PREFIXES):
                continue
            target = os.path.join(dest, name.replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def verify_zip(zip_path, offline):
    """Sanity-check a finished package (raises on problems)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        flat = [n.split("/", 1)[1] if "/" in n else n for n in names]

        for bad in ("release/", "dist/"):
            if any(n.startswith(bad) for n in flat if n):
                raise SystemExit(f"{os.path.basename(zip_path)}: 不应包含 {bad}")
        for bad in (".venv", "output", "webui_uploads"):
            if any(fn.startswith(bad + "/") for fn in flat):
                raise SystemExit(f"{os.path.basename(zip_path)}: 不应包含 {bad}/")
        if "使用说明.txt" not in flat:
            raise SystemExit(f"{os.path.basename(zip_path)}: 缺少 使用说明.txt")

        for rel in CMD_FILES:
            if rel not in flat:
                raise SystemExit(f"{os.path.basename(zip_path)}: 缺少 {rel}")
            head = names[flat.index(rel)]
            text = zf.read(head).decode("utf-8")
            has = any(
                ln.strip() == "set AUDIOSR_OFFLINE=1" for ln in text.splitlines()
            )
            if has != offline:
                raise SystemExit(
                    f"{os.path.basename(zip_path)}: {rel} 的离线标志与预期不符"
                )

        # the CLI entry point must propagate the python exit code
        entry = zf.read(names[flat.index(CMD_FILES[0])]).decode("utf-8")
        if "endlocal & exit /b" not in entry:
            raise SystemExit(
                f"{os.path.basename(zip_path)}: {CMD_FILES[0]} 未传递退出码"
            )
    return len(names)


def build_zip(src_dir, zip_path, root_name):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            dirnames.sort()
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, src_dir).replace(os.sep, "/")
                zf.write(full, f"{root_name}/{rel}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    version = args[0] if args else "v1.0.0"
    ref = args[1] if len(args) > 1 else "HEAD"
    with_model = "--with-model" in flags
    with_offline = "--with-offline" in flags
    variants = [ONLINE_VARIANT] + ([OFFLINE_VARIANT] if with_offline else [])
    os.makedirs(DIST_DIR, exist_ok=True)

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print(f"[warn] 工作区有未提交改动，包内容来自 {ref}（已提交部分）")

    tmp = tempfile.mkdtemp(prefix="audiosr_rel_")
    try:
        src_zip = os.path.join(tmp, "src.zip")
        subprocess.run(
            ["git", "archive", "--format=zip", "-o", src_zip, ref], cwd=ROOT, check=True
        )
        base = os.path.join(tmp, "src")
        os.makedirs(base)
        extract(src_zip, base)

        results = []
        for variant in variants:
            pkg_name = f"AudioSR-LowVRAM-{version}-{variant['tag']}"
            work = os.path.join(tmp, pkg_name)
            shutil.copytree(base, work)

            for rel in CMD_FILES:
                path = os.path.join(work, rel)
                if not os.path.exists(path):
                    raise SystemExit(f"缺少预期文件：{rel}")
                crlf(path)
                if variant["offline"]:
                    patch_offline(path)

            doc = with_version(
                open(os.path.join(RELEASE_DIR, variant["doc"]), encoding="utf-8").read(),
                version,
            )
            payload = b"\xef\xbb\xbf" + doc.replace("\n", "\r\n").encode("utf-8")
            open(os.path.join(work, "使用说明.txt"), "wb").write(payload)

            # ZIP 内部用中文文件夹名（Windows 资源管理器能正确解压），
            # 对外文件名用 ASCII，避开部分工具链对非 ASCII 文件名的丢失问题。
            asset_name = f"AudioSR-LowVRAM-{version}-{variant['slug']}.zip"
            out_zip = os.path.join(DIST_DIR, asset_name)
            build_zip(work, out_zip, pkg_name)
            entries = verify_zip(out_zip, variant["offline"])
            size_mb = os.path.getsize(out_zip) / 2 ** 20
            digest = sha256(out_zip)
            results.append((asset_name, size_mb, digest))
            print(
                f"[ok] {asset_name}  {size_mb:.1f} MB  {entries} 个文件  "
                f"（包内目录：{pkg_name}）  sha256={digest}"
            )

        body_src = os.path.join(RELEASE_DIR, f"发布说明_{version}.md")
        body = open(body_src, encoding="utf-8").read()
        table = "\n".join(f"- `{n}` — {s:.1f} MB\n  `{h}`" for n, s, h in results)
        body = body.replace("<!--CHECKSUMS-->", table)

        if with_model:
            model = find_model()
            if not model:
                raise SystemExit("找不到 pytorch_model.bin（可用 AUDIOSR_CKPT 指定）")
            digest = sha256(model)
            if digest != MODEL_SHA256:
                raise SystemExit(
                    "模型 SHA-256 与代码里的常量不一致（防止发布出打不开的包）：\n"
                    f"  文件: {digest}\n"
                    f"  常量: {MODEL_SHA256}\n"
                    "请同步 run_lowvram.BASIC_MODEL_SHA256 与本文件的 MODEL_SHA256。"
                )
            print(f"[info] 切分模型：{model}（SHA-256 与代码常量一致）")
            parts = split_model(model, DIST_DIR)
            lines = [
                f"完整文件 SHA-256：`{MODEL_SHA256}`（解压合并后会自动校验）",
                "",
                "| 分卷 | 大小 | SHA-256 |",
                "| --- | --- | --- |",
            ]
            lines += [f"| `{n}` | {s:.0f} MB | `{h}` |" for n, s, h in parts]
            body = body.replace("<!--MODELPARTS-->", "\n".join(lines))
        else:
            body = body.replace("<!--MODELPARTS-->", "（本次未生成模型分卷）")

        body_path = os.path.join(DIST_DIR, "release_body.md")
        open(body_path, "w", encoding="utf-8").write(body)
        print(f"[ok] {body_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
