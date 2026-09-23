# AGENTS.md

## 项目定位

FeiFei-Mini-Tools 是随手小工具集：**单文件、能跑、轻文档**。不要引入包结构、setup.py、poetry、CI 之类重型脚手架，除非用户明确要求。

## 环境与命令

```bash
# 依赖装进项目 venv（本仓库 .venv 由 uv 创建，用 uv pip）
uv pip install -r requirements.txt --python .venv/bin/python

# 跑工具 / 语法检查
.venv/bin/python <tool>.py
python3 -m py_compile <tool>.py
```

- 没有测试框架、没有 lint 配置；改完至少做 `py_compile`，涉及 GUI/线程的改动尽量用最小脚本实测退出路径。
- 不要擅自 `pip install` 新包进系统 Python；装进 `.venv` 并更新 `requirements.txt`。

## 代码约定

- 新工具：根目录一个 `xxx_yyy.py`，带 `if __name__ == "__main__":`。
- 注释和 UI 文案用中文；标识符用英文。
- 需要 `print` 进度/错误时加 `flush=True`（GUI 方式启动时方便看终端）。
- 第三方告警默认屏蔽，但**自己的错误必须露出来**（打到终端 + UI/日志）。

## GUI / 线程（踩过的坑，别再踩）

1. **不要给顶层窗口加 `Qt.WindowType.SubWindow` 或 `Qt.WindowType.Tool`**：关闭后不触发 `lastWindowClosed`，`app.exec()` 永不返回，表现为点退出假死。只用 `FramelessWindowHint | WindowStaysOnTopHint` 即可。
2. **`closeEvent` 里禁止长时间 `QThread.wait()` / `terminate()`**：只置停止标志；线程回收放到 `app.exec()` 返回之后，超时用 `os._exit()` 兜底。
3. **QThread.run 全程 try/except/finally**：finally 里释放模型 / `torch.cuda.empty_cache()`；循环内每步昂贵调用（ASR、翻译）单独 try，一处失败不能弄死整个线程。
4. **跨线程更新 UI 只走 signal**，不要在 worker 里直接碰控件。
5. 硬编码设备号、pad token id、魔法阈值 → 改成常量 / 环境变量 / tokenizer 字段。

## 改动后的最低验证

```bash
python3 -m py_compile <改动的文件>.py
```

涉及窗口关闭：写 20 行左右的最小复现脚本，确认 `app.exec()` 能在 1s 内返回。

## 文档

- `README.md`：给使用者看——装依赖、怎么跑、工具表格一行一个。
- 依赖版本写进 `requirements.txt`（下限用 `>=`，实测过的版本即可）。
- 不要生成多余的 md 文件。
