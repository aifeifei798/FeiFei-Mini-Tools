# FeiFei-Mini-Tools

随手攒的小工具集，单文件、能直接跑就行，不追求工程化。

## 环境

```bash
# 推荐（本仓库 .venv 由 uv 创建）
uv venv
uv pip install -r requirements.txt

# 或标准 venv
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- `live_subtitles.py` 需要 NVIDIA GPU（CUDA）；麦克风默认用系统输入设备，可用 `LIVE_SUBTITLES_MIC` 指定设备号。
- `download_organizer.py` 可用 `--device cpu` 无 GPU 跑；首次运行从 HuggingFace 拉模型。

## 工具

### live_subtitles.py — 实时同传悬浮窗

麦克风实时拾音 → Qwen3-ASR 识别 → Hy-MT2 多语翻译 → 无边框置顶悬浮字幕。

```bash
.venv/bin/python live_subtitles.py
```

- 顶部：`识别`/`译文` 语言下拉，字号下拉（9~48px）+ `A-`/`A+` 微调，`清空` 清历史，`✕` 或 `Esc` 退出
- 语言默认 `auto → auto`：识别侧 auto = 模型自动判语种；译文侧 auto = 系统语言（如 zh_CN → 中文），也可手选（如 auto → 中文）
- 按住窗口任意处拖动；右下角手柄缩放；窗口默认尺寸按屏幕比例（1080p 收敛，4K 放大）
- 底部蓝字为实时识别原文（随字号缩放），上方大框为译文历史（最新一句金色）

首次运行会从 HuggingFace 下载模型：

- `Qwen/Qwen3-ASR-0.6B`
- `tencent/Hy-MT2-1.8B`

### download_organizer.py — 下载目录智能归档

先加载 Laya 模型 → 扫描文件并分类（进度条）→ 移入 `~/Archive/0X_xxx/`（进度条）。  
默认**只预览不移动**，确认后再 `--apply`。

```bash
# 预览（不动文件，只扫顶层）
.venv/bin/python download_organizer.py

# 递归扫子目录
.venv/bin/python download_organizer.py -r

# 真正归档
.venv/bin/python download_organizer.py --apply -r

# 常用参数
#   -r, --recursive      递归扫描子目录（默认只扫顶层）
#   --path DIR           待整理目录（默认 ~/Downloads）
#   --device cuda|cpu    推理设备（默认自动，无 GPU 写 cpu）
#   --min-confidence 0.5 置信度阈值 0~1，默认 0.5；写 0 = 全部都归档
```

**置信度怎么调**：表格里「置信度」列低于 `--min-confidence` 的文件会跳过（只提示、不移动）。  
例如默认 0.5 时某文件是 0.42 被跳过，想收进来就改成更低的值：

```bash
# 只收 0.3 以上的
.venv/bin/python download_organizer.py --min-confidence 0.3

# 不设门槛，全部归档
.venv/bin/python download_organizer.py --min-confidence 0
```

脚本结束时若有跳过，会直接打印可复制的示例命令。

归档根目录固定为 `~/Archive/`，六个分类子目录见脚本内 `CATEGORIES`。  
`-r` 扫到的子目录文件会**打平**移入分类目录（不保留原子目录结构）。  
重名文件自动加时间戳后缀，**绝不覆盖**已归档文件。  
忽略 `.crdownload` / `.part` / `.aria2` 等下载中临时文件与隐藏文件（含路径中任意 `.` 开头的目录）。  
输出顺序固定：模型加载日志 → 「扫描到 N 个文件」→ 分类表 + 进度条，互不抢行。

## 加新工具

- 一个工具一个 `xxx_yyy.py`，能 `python xxx_yyy.py` 直接跑
- 公共依赖补进 `requirements.txt`
- 在下面表格加一行即可，不用写长文档

| 工具 | 说明 |
|------|------|
| `live_subtitles.py` | 实时语音同传悬浮字幕（识别/译文语言可选，字号 9~48px 可选，auto→auto） |
| `download_organizer.py` | 下载目录语义分类归档到 ~/Archive（支持 `-r` 递归） |
