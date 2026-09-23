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
- 新依赖直接 `import` 的包必须写进 `requirements.txt`（即使用的是传递依赖，如 `tqdm`）。

## CLI 约定（批量文件类工具）

1. **破坏性操作默认关闭**：先预览，真执行用 `--apply`（或同类开关）。
2. **错误必须非 0 退出**：目录不存在、模型加载失败、全部任务失败 → `sys.exit(1)`；成功才 0。
3. **绝不覆盖用户文件**：目标已存在则循环改名（`stem_mtime`、`stem_mtime_1`…），用 `while dest.exists()` 而不是改一次就 `shutil.move`。
4. **单条记录失败不中断整批**：`predict` / `move` 各自 try，记入 failed 列表，结束时汇总打印。
5. **进度条与表格分行**：`tqdm` 走 `sys.stderr`；表格和警告用 `tqdm.write`（或 stdout），不要和 bar 抢同一行。
6. **输出顺序**：先完成重活（加载模型、HF 下载），再打印「扫描到 N 个…」和结果表；不要在分类中途才第一次拉模型。
7. **提示要给可复制命令**：例如跳过低置信度时，打印具体的 `--min-confidence 0` / `--min-confidence 0.3` 示例，不要只说「调低阈值后重试」。
8. **阈值/设备/路径做成参数**：默认值合理，help 里带例子；不要硬编码 `device="cuda"`、魔法阈值。

## GUI / 线程（踩过的坑，别再踩）

1. **不要给顶层窗口加 `Qt.WindowType.SubWindow` 或 `Qt.WindowType.Tool`**：关闭后不触发 `lastWindowClosed`，`app.exec()` 永不返回，表现为点退出假死。只用 `FramelessWindowHint | WindowStaysOnTopHint` 即可。
2. **`closeEvent` 里禁止长时间 `QThread.wait()` / `terminate()`**：只置停止标志；线程回收放到 `app.exec()` 返回之后，超时用 `os._exit()` 兜底。
3. **QThread.run 全程 try/except/finally**：finally 里释放模型 / `torch.cuda.empty_cache()`；循环内每步昂贵调用（ASR、翻译）单独 try，一处失败不能弄死整个线程。
4. **跨线程更新 UI 只走 signal**，不要在 worker 里直接碰控件。
5. 硬编码设备号、pad token id、魔法阈值 → 改成常量 / 环境变量 / tokenizer 字段 / CLI 参数。

## 改动后的最低验证

```bash
python3 -m py_compile <改动的文件>.py
```

- 涉及窗口关闭：写 20 行左右的最小复现脚本，确认 `app.exec()` 能在 1s 内返回。
- 涉及扫描/过滤：用 `mktemp -d` 造顶层 + 子目录 + 隐藏目录 + `.part` 文件，确认入选集合正确（可不联网，把模型加载 stub 掉）。
- 涉及参数提示：跑 `--help` 和一次干跑，确认文案里有具体数字/命令示例。

## 文档

- `README.md`：给使用者看——装依赖、怎么跑、工具表格一行一个；参数示例要能直接复制。
- 依赖版本写进 `requirements.txt`（下限用 `>=`，实测过的版本即可）。
- 不要生成多余的 md 文件；改工具行为时同步改 README 对应段落和 `--help`。
