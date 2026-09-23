# FeiFei-Mini-Tools

随手攒的小工具集，单文件、能直接跑就行，不追求工程化。

## 环境

```bash
# 推荐（本仓库 .venv 是 uv 建的）
uv venv
uv pip install -r requirements.txt

# 或标准 venv
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

需要 NVIDIA GPU（CUDA）；麦克风用系统默认输入设备，可用环境变量指定：

```bash
export LIVE_SUBTITLES_MIC=5   # 可选，默认不设 = 系统 default
```

## 工具

### live_subtitles.py — 5090 实时同传悬浮窗

麦克风实时拾音 → Qwen3-ASR 识别 → Hy-MT 中英互译 → 无边框置顶悬浮字幕。

```bash
.venv/bin/python live_subtitles.py
```

- 顶部：`A-`/`A+` 调字号，`清空` 清历史，`✕` 或 `Esc` 退出
- 按住窗口任意处拖动；右下角手柄缩放
- 底部蓝字为实时识别原文，上方大框为译文历史（最新一句金色）

首次运行会从 HuggingFace 下载模型：

- `Qwen/Qwen3-ASR-0.6B`
- `tencent/Hy-MT2-1.8B`

## 加新工具

- 一个工具一个 `xxx_yyy.py`，能 `python xxx_yyy.py` 直接跑
- 公共依赖补进 `requirements.txt`
- 在下面表格加一行即可，不用写长文档

| 工具 | 说明 |
|------|------|
| `live_subtitles.py` | 实时语音同传悬浮字幕 |
