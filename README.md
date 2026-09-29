# AudioSR 低显存版 · 中文说明（4 GB 显存可跑 + 本地网页界面）

> 🌐 **中文（当前页）** · [English / 上游英文文档 →](README_EN.md)

> ### ⚠️ 来源与声明
>
> 本仓库是 **[haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution)**（项目名 **AudioSR**，作者 Haohe Liu、Ke Chen、Qiao Tian、Wenwu Wang、Mark D. Plumbley）的**第三方修改版**。
>
> - 核心算法与模型代码（`audiosr/` 目录）、示例音频、论文成果均来自上游，**版权归原作者所有**；上游以 **MIT 许可证**发布，本仓库**原样保留** `LICENSE` 文件。
> - 本仓库**没有修改**上游的模型结构或训练/推理算法，只增加了外围工具：低显存推理封装、本地网页界面（含局域网访问）、Windows 双击脚本、自检脚本。
> - 若这些改动对你有帮助，请**优先给上游项目点 Star**；用于研究或论文时请引用上游工作（见文末 [引用](#引用上游工作)）。
>
> English note: this repository is a modified community version of AudioSR (MIT). See `NOTICE` for details. The upstream English README is preserved in [`README_EN.md`](README_EN.md).

---

## 目录

1. [这个版本解决了什么](#1-这个版本解决了什么)
2. [新增文件清单](#2-新增文件清单)
3. [环境要求与安装](#3-环境要求与安装)
4. [四种使用方式](#4-四种使用方式)
5. [输入与输出](#5-输入与输出)
6. [低显存原理与实测数据](#6-低显存原理与实测数据)
7. [命令行参数](#7-命令行参数)
8. [自检脚本](#8-自检脚本)
9. [常见问题](#9-常见问题)
10. [发行版（Releases）](#10-发行版releases)
11. [许可与致谢](#11-许可与致谢)

---

## 1. 这个版本解决了什么

上游 AudioSR 默认把整套模型（UNet + 两套 VAE + HiFiGAN 声码器，fp32 合计约 5.7 GB）放在同一张显卡上，4 GB 显存的笔记本显卡必然 OOM。

本版本的做法：

- **只把 fp16 的 UNet 放显卡**（约 1.5 GB），两套 VAE 与声码器留在 CPU；
- 注意力替换为 PyTorch 的 **SDPA**（Flash/mem-efficient 内核），避免实例化 `(b·h, N, N)` 注意力矩阵；
- UNet 前向包在 `torch.autocast(float16)` 里，混合精度安全；
- 可选 `--gpu_vae`：把 VAE + 声码器也搬到显卡（fp16），更快，但显存占用更高；
- 结果：**RTX 3050 Laptop（4 GB）可以稳定推理**，实测见 [第 6 节](#6-低显存原理与实测数据)。

另外补齐了上游缺的易用性：网页界面（含手机可用）、双击即用脚本、长音频分块、真立体声逐声道处理、多输出格式/采样率。

## 2. 新增文件清单

| 文件                                    | 说明                                                                                                                          |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `run_lowvram.py`                        | 低显存推理核心：fp16 UNet 上显卡、SDPA 补丁、显存优化版 `generate_batch`、长音频分块、真立体声、重采样与多格式导出            |
| `webui_server.py`                       | 本地网页界面（仅用 Python 标准库 + 项目依赖，无需 Gradio）：上传 → 处理 → 进度 → 试听 → 下载/打开文件夹；模型常驻，只加载一次 |
| `双击启动网页界面.cmd`                  | 一键启动网页界面（仅本机，`127.0.0.1:7860`）                                                                                  |
| `双击启动局域网网页界面.cmd`            | 一键启动网页界面并监听所有网卡（`0.0.0.0`），手机/同网电脑可访问                                                              |
| `双击选择音频处理.cmd`                  | 弹窗选择音频 → 默认参数处理 → 自动打开输出文件夹（调用 `pick_and_run.py`）                                                    |
| `pick_and_run.py`                       | 上面的弹窗逻辑与默认参数                                                                                                      |
| `run_audiosr.cmd`                       | 命令行入口，自动带上 `.venv` 与本地权重                                                                                       |
| `test_formats.py`                       | 自检：mp3/flac/m4a/ogg 导出是否合法（ffprobe 校验）                                                                           |
| `test_192k.py`                          | 自检：192 kHz 导出容器正确、>24 kHz 无新增内容                                                                                |
| `test_outsr.py`                         | 自检：输出采样率重采样与立体声左右平衡                                                                                        |
| `check_output.py`                       | 自检：输入/输出高频能量对比 + 权重参数分布                                                                                    |
| `README.md` / `README_EN.md` / `NOTICE` | 中文说明（仓库首页）、上游英文文档与来源声明                                                                                  |

> 未包含在上游原仓库中的还有：`.venv/`（虚拟环境）、模型权重、`output*/` 生成音频、`webui_uploads/` 上传缓存 —— 这些已在 `.gitignore` 中排除。

## 3. 环境要求与安装

**实测环境**：Windows 11 + NVIDIA RTX 3050 Laptop 4 GB + Python 3.10.1 + PyTorch 2.0.1+cu118 / torchaudio 2.0.2+cu118。

- **Python 3.9 / 3.10 / 3.11（推荐 3.10）**：`torch==2.0.1+cu118` 没有 3.12+ 的安装包，用 3.12/3.13/3.15 会报 `Could not find a version that satisfies the requirement torch==2.0.1+cu118`。纯 CPU 也能跑（`--device cpu`，只是慢很多）。
- **`ffmpeg` / `ffprobe` 必需**：上游的保存流程（`audiosr/utils.py` 的 `save_wave → strip_silence`）会调用它们把输出裁剪到原音频时长；处理 `m4a/aac/wma` 等输入或导出 mp3/flac/m4a/ogg 也靠它。Windows 可 `winget install Gyan.FFmpeg`，并确保 `ffmpeg`、`ffprobe` 在 `PATH` 中。
- 模型权重查找顺序（见 `run_lowvram.py` 的 `default_ckpt_path()`）：
  1. 环境变量 `AUDIOSR_CKPT`；
  2. 程序目录 `pytorch_model.bin`（没有时先尝试合并 `pytorch_model.bin.part*` 分卷）；
  3. `models\pytorch_model.bin`；
  4. 上一级目录的 `pytorch_model.bin`；
  5. 都没有时，自动从 Hugging Face 下载（`haoheliu/audiosr_basic` / `haoheliu/audiosr_speech`）。
     国内网络会自动走镜像 `HF_ENDPOINT=https://hf-mirror.com`（脚本已内置），失败会自动再试官方站；
     同时设置 `NO_PROXY=*` 绕过系统代理。
- **所有下载都在程序目录，不占 C 盘**：`HF_HOME`/`HF_HUB_CACHE` 默认指向 `程序目录\models\`；
  安装依赖时 pip 缓存与临时文件用 `程序目录\.pip-cache\`、`程序目录\.tmp\`（安装完可删）。
  另外，早期版本会在导入时下载 `roberta-base`（约 500 MB），现已改为惰性加载、无需下载。

安装依赖（推荐双击 **`安装依赖.cmd`**，会自动选兼容的 Python 并创建 `.venv`）：

```shell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> 安装**不需要访问 GitHub**：上游 `requirements.txt` 里那条 `git+…/diffusers.git` 已注释（本项目没有代码用到它）。
> 所有 `.cmd` 脚本会自动寻找 `.venv`，其次是兼容的系统 Python（3.9~3.11）；直接用 `python run_lowvram.py` 时请在仓库根目录执行（脚本依赖同目录的本地 `audiosr/` 包）。

## 4. 四种使用方式

### 4.1 网页界面（本机）

双击 **`双击启动网页界面.cmd`** → 浏览器自动打开 <http://127.0.0.1:7860>。

- 拖拽或点选音频 → 可调选项：采样步数（10–200，默认 50）、随机种子（带“随机”按钮）、引导强度（1–10，默认 3.5）、分块时长（5–30 秒，默认 10）、输出采样率（48 kHz 原生 / 44.1 / 96 / 32 / 24 / 16 / 192 kHz）、输出格式（wav / flac / mp3 / m4a / ogg）、双声道处理（逐声道真立体声 / 混合单声道）、GPU 加速开关（默认勾选，VAE+声码器上显卡，峰值显存约 2.9 GB；显存吃紧时可取消）。
- 处理过程有状态徽章、实时日志与进度条；完成后可在页面直接试听、下载结果、或用“打开输出文件夹”定位到 `output\`。
- 页面底部还有一段“选项说明”，解释了每个选项的取舍。
- 模型在后台预热（首次约 40 秒），页面顶部会显示“模型加载中 / 模型已就绪”；加载完成后每次处理都复用同一份模型。
- 若 7860 端口被占用，会自动 +1 重试（最多 5 个端口）；实际端口以窗口打印为准。

### 4.2 网页界面（局域网 / 手机、平板、其它电脑）

双击 **`双击启动局域网网页界面.cmd`**，窗口会打印本机地址，例如：

```
[webui] serving at http://127.0.0.1:7860  (Ctrl+C to stop)
[webui] LAN access from phones / other PCs on the same network:
[webui]   http://192.168.5.13:7860
```

让手机连**同一个 Wi-Fi**，浏览器打开打印出的 `http://<本机IP>:7860` 即可上传并处理音频（输出文件仍保存在电脑的 `output\` 里）。

注意事项：

- 首次启动时 Windows 防火墙弹窗请选择**允许**（至少勾选“专用网络”）；若误点拒绝，可在“Windows 防火墙 → 允许应用或功能通过 Windows 防火墙”中为 Python 勾选专用网络。
- 局域网模式**没有任何鉴权**，请只在可信网络（家里/办公室内网）使用，用完按 `Ctrl+C` 关闭窗口。
- 关闭窗口即停止服务。

### 4.3 双击选择文件处理

双击 **`双击选择音频处理.cmd`** → 弹窗选择音频（支持 wav/flac/mp3/ogg/m4a/aac/wma）→ 使用默认参数（50 步、seed 42、guidance 3.5、自动分块、立体声逐声道）处理 → 完成后自动打开输出文件夹。

### 4.4 命令行

```shell
rem 短文件（≤12 秒单次推理）
run_audiosr.cmd -i "D:\audio\clip.wav"

rem 长音频：强制分块 + 每块 10 秒、重叠 2 秒，并额外导出 mp3
run_audiosr.cmd -i "D:\audio\song.mp3" --chunking --chunk_duration 10 --out_format mp3

rem 更快：VAE/声码器也上显卡（显存占用更高）
run_audiosr.cmd -i "D:\audio\clip.wav" --gpu_vae
```

跨平台（Linux/macOS）等价写法：

```shell
python run_lowvram.py -i clip.wav --out_sr 48000 --stereo split
```

## 5. 输入与输出

- **输入**：`wav / flac / mp3 / ogg / opus` 直接支持；`m4a / aac / wma` 等会自动用 ffmpeg 转成 wav；单声道、立体声均可（立体声默认逐声道处理，保持真实左右声像）。
- **输出**：`output\<时间戳>\<原文件名><后缀>.wav`
  - 默认后缀 `_AudioSR_Processed_48K`；改了 `--out_sr` 会自动改成对应数值（如 `_AudioSR_Processed_44K`，192 kHz 为 `_192K`）。
  - 加 `--out_format mp3|flac|m4a|ogg` 时会**额外**导出该格式，wav 母带始终保留。
  - 网页界面的处理结果同样落在 `output\<时间戳>\`。

## 6. 低显存原理与实测数据

| 做法                                                                  | 效果                                              |
| --------------------------------------------------------------------- | ------------------------------------------------- |
| 先在 CPU 上 fp32 构建整套 pipeline，再把 fp16 UNet 移到显卡           | 只在显卡上保留必要权重，约 1.5 GB                 |
| 两套 VAE + VAE 解码器 + HiFiGAN 声码器留在 CPU                        | 省下数百 MB ~ 1 GB 显存（`--gpu_vae` 可切回 GPU） |
| 注意力替换为 `scaled_dot_product_attention`                           | 避免 `(b·h, N, N)` 注意力矩阵，长音频不炸显存     |
| UNet 前向 `torch.autocast(float16)` + `apply_model` 包一层并转回 fp32 | 免改模型代码即得混合精度                          |
| 调度器 buffer（betas/alphas_cumprod 等）留在显卡，避免每步搬运        | 少了一次 PCIe 往返                                |
| 长音频分块（默认 >12 秒自动开启，10 秒/块、2 秒重叠）                 | 峰值显存与音频时长基本解耦                        |

**实测数据**（RTX 3050 Laptop 4 GB，输入 `example/sound_effect.wav`：10.24 秒、48 kHz、单声道，50 步 DDIM，seed 42）：

| 配置                                  | DDIM 50 步 | 峰值显存                     | 端到端耗时 |
| ------------------------------------- | ---------- | ---------------------------- | ---------- |
| 默认（VAE/声码器在 CPU）              | 8.6 s      | **0.63 GiB**（仅 DDIM 阶段） | 51.0 s     |
| `--gpu_vae`（VAE/声码器上显卡，fp16） | 7.6 s      | **2.93 GiB**（含 VAE 解码）  | **19.2 s** |

结论：4 GB 显存两种配置都跑得下；默认配置给显存留足余量，`--gpu_vae` 约快 2.7 倍。长音频按音频时长线性增长（分块后每块独立推理），参考：网页界面提示“GPU 加速下单声道约为音频时长的 2–3 倍，立体声逐声道约 4–6 倍”。

## 7. 命令行参数

| 参数                                      | 默认       | 说明                                                                   |
| ----------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| `-i`, `--input_audio_file`                | 必填       | 输入音频                                                               |
| `-s`, `--save_path`                       | `./output` | 输出根目录（自动建时间戳子目录）                                       |
| `--model_name`                            | `basic`    | `basic`（音乐/音效）或 `speech`（语音）                                |
| `--ddim_steps`                            | `50`       | 采样步数，越大越慢                                                     |
| `-gs`, `--guidance_scale`                 | `3.5`      | 指导系数                                                               |
| `--seed`                                  | `42`       | 随机种子                                                               |
| `--out_sr`                                | `48000`    | 8000~192000；模型内容上限约 24 kHz，导出 >48 kHz 不会新增高频内容      |
| `--out_format`                            | `wav`      | `wav/mp3/flac/m4a/ogg`（需 ffmpeg；wav 母带保留）                      |
| `--chunking`                              | 关         | 强制分块；不加则 >12 秒自动分块                                        |
| `--chunk_duration` / `--overlap_duration` | `10` / `2` | 分块时长与重叠（秒）                                                   |
| `--stereo`                                | `split`    | `split` = 逐声道真立体声（约 2 倍耗时）；`mono` = 旧行为（单声道下混） |
| `--gpu_vae`                               | 关         | VAE/声码器上显卡（fp16），更快但更吃显存                               |
| `--device`                                | `cuda:0`   | 也可 `cpu`                                                             |
| `--ckpt_path`                             | 见第 3 节  | 手动指定权重；也可用环境变量 `AUDIOSR_CKPT`                            |

网页界面里的选项与上表一一对应（输出采样率、格式、分块、立体声、步数、种子、指导系数）。

## 8. 自检脚本

```shell
.venv\Scripts\python.exe test_formats.py   # 四种输出格式是否合法（需 ffmpeg/ffprobe）
.venv\Scripts\python.exe test_192k.py      # 192 kHz 导出与 >24 kHz 空频谱验证
.venv\Scripts\python.exe test_outsr.py     # 44.1 kHz 重采样与立体声平衡
.venv\Scripts\python.exe check_output.py example\music.wav output\<时间戳>\xxx_48K.wav
```

## 9. 常见问题

**Q：显存溢出（CUDA out of memory）？**
不要加 `--gpu_vae`；关掉其它占用显卡的程序（浏览器视频、游戏）；确认日志里有 `[lowvram] DDIM ... peak VRAM x.xx GiB`，说明低显存分支生效。

**Q：报“不支持的音频格式，且系统里找不到 ffmpeg” / 报找不到 `ffprobe`？**
`ffmpeg` 与 `ffprobe` 都是**必需**的（保存输出时会调用它们裁剪时长）。安装 [ffmpeg](https://ffmpeg.org/download.html)（Windows 可 `winget install Gyan.FFmpeg`）并确保两者都在 `PATH` 中；只支持 wav/flac/mp3/ogg/opus 作为直接输入，`m4a/aac/wma` 也需要它来转换。

**Q：模型下载卡住/失败？**
设置 `AUDIOSR_CKPT` 指向已下载的 `pytorch_model.bin`（例如权重放在仓库上一级目录即可被自动识别），或保持 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像。

**Q：导出 96 kHz/192 kHz，但频谱 24 kHz 以上是空的？**
正常现象：AudioSR 的内容上限约 24 kHz（模型在 48 kHz 下工作），“更高采样率”只是重采样容器，`test_192k.py` 专门验证了这一点。

**Q：处理一首 3 分钟的歌很慢？**
默认 >12 秒即自动分块，RTX 3050（4 GB）上大致是“每 10 秒音频几十秒”的量级；可减小 `--chunk_duration`、降低 `--ddim_steps`，或加 `--gpu_vae` 提速。

**Q：立体声为什么耗时翻倍？**
`--stereo split` 会把左右声道分别过一遍模型（换取真实声像）；追求速度可改 `--stereo mono`。

**Q：网页打不开 / 手机连不上？**
端口占用会自动 +1，注意看窗口打印的实际端口；手机连不上多半是防火墙未放行（见 4.2），或手机与电脑不在同一网段（访客 Wi-Fi 常见）。

**Q：`webui_uploads\` 和 `output\` 可以删吗？**
可以，都是普通结果文件；两者已在 `.gitignore` 中，不会进版本库。

**Q：怎样彻底禁止程序联网下载模型？**
设置环境变量 `AUDIOSR_OFFLINE=1`（在启动脚本里加一行 `set AUDIOSR_OFFLINE=1` 即可）：此时只使用本地权重，找不到就直接报错，不会发起任何下载。

**Q：双击 .cmd 提示找不到 Python 环境？**
启动脚本会先找同目录的 `.venv`，再找系统 Python（需已装依赖）。都没有时：双击 **`安装依赖.cmd`**（联网一次，自动创建 `.venv`）；也可以把有网机器上装好的整个文件夹（含 `.venv`）拷过来。

**Q：安装依赖报 `Could not find a version that satisfies the requirement torch==2.0.1+cu118`？**
你的 Python 是 3.12 或更高（该版 torch 没有对应安装包）。安装 Python 3.10 后重跑 `安装依赖.cmd`，或 `set AUDIOSR_PY_BASE=D:\Python310\python.exe` 指定已有解释器。

**Q：下载的模型和缓存会占用 C 盘吗？**
不会。`HF_HOME`/`HF_HUB_CACHE` 默认指向 `程序目录\models\`，pip 缓存/临时文件用 `程序目录\.pip-cache\`、`程序目录\.tmp\`；上传与输出也在程序目录。安装完成后可自由删除 `.pip-cache`、`.tmp` 释放空间。

**Q：安装依赖卡在 `git clone … huggingface/diffusers` 报 `Recv failure: Connection was reset`？**
这是国内访问 GitHub 被重置。该依赖本项目并未用到，已在 `requirements.txt` 中注释掉；用最新版包的 `安装依赖.cmd` 即可正常安装。

## 10. 发行版（Releases）

不想自己配环境？[Releases](https://github.com/Simlalsy/audiosr-4gb-webui/releases) 提供打包好的 Windows 包（解压即用）：

| 包                                                   | 说明                                                                                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `AudioSR-LowVRAM-v1.1.4-A-online-model-download.zip` | 首次运行自动从 Hugging Face 镜像下载 `pytorch_model.bin`（约 5.75 GB，保存到**程序目录** `models\`，不占 C 盘）；程序目录已有完整模型时不会下载 |

用法：解压 → 双击 **`安装依赖.cmd`**（联网一次，自动建 `.venv`）→ 双击 `双击启动网页界面.cmd`；首次启动自动下载权重（已放好完整模型则不会下载）。包内 `使用说明.txt` 有逐步说明。

想**完全离线**运行：把 `pytorch_model.bin` 放到程序目录（或 `models\`，或用 `AUDIOSR_CKPT` 指定），并在启动脚本里加一行 `set AUDIOSR_OFFLINE=1`。

自己重新打包：在仓库根目录执行 `python release\build_release.py v1.1.4`（可选 `--with-offline` 生成完全离线版、`--with-model` 生成模型分卷），产物在 `dist\`，发布文案在 `release\发布说明_v1.1.4.md`。

> **权重查找顺序**（两版本通用）：`AUDIOSR_CKPT` → 程序目录 `pytorch_model.bin`（没有时先尝试合并 `*.part*` 分卷）→ `models\pytorch_model.bin` → 上一级目录 `pytorch_model.bin`；都没找到时才联网下载（设了 `AUDIOSR_OFFLINE=1` 则直接报错）。

## 11. 许可与致谢

- 上游项目 **AudioSR**：MIT License，作者 Haohe Liu 等，见 [LICENSE](LICENSE)（原样保留）与 [NOTICE](NOTICE)。
- 本仓库的新增代码同样以 MIT 许可发布；示例音频、图片等资源版权归上游作者。
- 请勿将本工具用于侵犯他人版权的用途（如对受版权保护的录音做再发布）。

### 引用上游工作

```bibtex
@inproceedings{liu2024audiosr,
  title={{AudioSR}: Versatile audio super-resolution at scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={1076--1080},
  year={2024},
  organization={IEEE}
}
```
