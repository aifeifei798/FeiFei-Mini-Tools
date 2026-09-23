import argparse
import os
from pathlib import Path
import shutil
import sys
from laya import Router
from tqdm import tqdm

# ================= 1. 分类目标与规则定义 =================
TARGET_BASE_DIR = Path.home() / "Archive"

CATEGORIES = {
    "rc_engines": {
        "folder": "00_航模涡喷与机械图纸",
        "desc": (
            "航模、涡轮喷气发动机、脉冲发动机制作手册、机械图纸CAD、图纸RAR、SolidWorks模型"
        ),
    },
    "invoices": {
        "folder": "01_发票账单",
        "desc": "电子发票、滴滴行程单、付款收据、报销单、消费明细",
    },
    "papers_docs": {
        "folder": "02_论文与书籍",
        "desc": "学术论文(arXiv/PDF)、电子书、技术白皮书、行业研报、长篇文档",
    },
    "installers": {
        "folder": "03_软件安装包",
        "desc": (
            "Linux安装包(deb/rpm/AppImage)、压缩包(zip/tar/gz)、驱动、固件"
        ),
    },
    "code_data": {
        "folder": "04_代码与数据",
        "desc": "代码源码(py/cpp/sh)、JSON/CSV/Parquet数据集、SQL备份、Patch补丁",
    },
    "media_subtitles": {
        "folder": "05_字幕与媒体",
        "desc": "视频字幕(srt/vtt/ass)、壁纸图片、短视频、录音音频",
    },
}

# 忽略正在下载的临时文件和隐藏文件
IGNORE_EXTS = {
    ".crdownload",
    ".part",
    ".tmp",
    ".download",
    ".aria2",
    ".partial",
    ".opendownload",
    ".!qb",
    ".bt!",
    ".torrent",
}


# ================= 2. 提取文件预览特征 =================
def get_file_snippet(file_path: Path) -> str:
    """提取文件前 300 字符作为语义特征，辅助 Laya 做深度理解"""
    ext = file_path.suffix.lower()

    # 针对文本类文件（字幕、代码、Markdown、CSV、脚本）
    text_exts = {
        ".txt",
        ".srt",
        ".vtt",
        ".py",
        ".sh",
        ".json",
        ".csv",
        ".md",
        ".log",
    }
    if ext in text_exts:
        try:
            with open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                return f.read(300).replace("\n", " ")
        except Exception:
            pass

    # 针对二进制大文件（安装包、压缩包），直接靠后缀和文件类型描述
    if ext in {".deb", ".rpm", ".appimage"}:
        return "Linux package binary installer"
    if ext in {".zip", ".tar", ".gz", ".xz", ".7z"}:
        return "Compressed archive file"
    if ext in {".mp4", ".mkv", ".mp3", ".wav"}:
        return "Media audio or video container"

    return "Document or binary data"


def unique_dest(folder: Path, src: Path) -> Path:
    """生成不与已有文件冲突的目标路径（循环查重，绝不覆盖）。"""
    dest = folder / src.name
    if not dest.exists():
        return dest
    stem, ext = src.stem, src.suffix
    mtime = int(src.stat().st_mtime)
    candidate = folder / f"{stem}_{mtime}{ext}"
    if not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = folder / f"{stem}_{mtime}_{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


# ================= 3. 主程序 =================
def main():
    parser = argparse.ArgumentParser(
        description="基于 Laya 的 Downloads 智能分类整理器"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行文件移动（默认仅模拟演练预览）",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(Path.home() / "Downloads"),
        help="待整理的目录，默认 ~/Downloads（默认仅顶层，加 -r 扫子目录）",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归扫描子目录中的文件（默认只扫顶层）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，如 cuda / cpu；默认自动选择",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help=(
            "置信度阈值 0~1，低于该值的文件不归档，只提示（默认 0.5）。"
            "例：--min-confidence 0.3 表示 0.30 及以上才归档；"
            "全要归档可 --min-confidence 0"
        ),
    )
    args = parser.parse_args()

    downloads_dir = Path(args.path).expanduser()
    if not downloads_dir.is_dir():
        print(f"❌ 目录不存在: {downloads_dir}", flush=True)
        sys.exit(1)

    try:
        target_base = TARGET_BASE_DIR.resolve()
    except Exception:
        target_base = TARGET_BASE_DIR

    # 先静默列目录，空目录不必加载模型
    files_to_process = []
    if args.recursive:
        candidates = (p for p in downloads_dir.rglob("*") if p.is_file())
    else:
        candidates = (p for p in downloads_dir.iterdir() if p.is_file())

    for f in candidates:
        try:
            rel_parts = f.relative_to(downloads_dir).parts
        except ValueError:
            continue
        # 跳过路径中任一以 . 开头的隐藏文件/目录
        if any(part.startswith(".") for part in rel_parts):
            continue
        if f.suffix.lower() in IGNORE_EXTS:
            continue
        # 已在归档目录内的文件不再搬动
        try:
            if f.resolve().is_relative_to(target_base):
                continue
        except Exception:
            pass
        files_to_process.append(f)

    if not files_to_process:
        print("🎉 下载目录干干净净，没有需要整理的文件！", flush=True)
        return

    # 再完整加载模型（HF 下载/构建只在这段出现），避免和扫描/分类输出抢行
    print(">>> 正在加载 Laya 决策引擎（首次运行需下载模型）...", flush=True)
    router = Router(preload=False, device=args.device)
    try:
        # 中英文件名都可能被路由到，先装常用两个检查点
        router.preload([router.default, "multilingual"])
    except Exception as e:
        print(f">>> 预载双语模型失败，回退默认模型: {e}", flush=True)
        try:
            router.preload([router.default])
        except Exception as e2:
            print(f"❌ Laya 模型加载失败: {e2}", flush=True)
            sys.exit(1)
    print(">>> Laya 加载完成\n", flush=True)

    # 构建 Laya 选择题
    questions = {
        "category": {
            "type": "choice",
            "instructions": "该文件最属于以下哪一类用途？",
            "criteria": {k: v["desc"] for k, v in CATEGORIES.items()},
        }
    }

    scope = "含子目录" if args.recursive else "仅顶层"
    print(
        f"🔍 扫描到 {len(files_to_process)} 个文件（{scope}），开始分类...\n",
        flush=True,
    )
    print("=" * 90, flush=True)
    print(f"{'文件名':<36} {'分类结果':<16} {'置信度':<8} {'目标归档路径'}", flush=True)
    print("=" * 90, flush=True)

    moves = []       # (src, dest, dest_dir)
    skipped = []     # (name, conf, folder)
    failed = []      # (name, err)

    # 进度条走 stderr，表格走 stdout，互不抢行
    bar = tqdm(
        files_to_process,
        desc="扫描分类",
        unit="file",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for file_path in bar:
        bar.set_postfix_str(file_path.name[:32], refresh=False)
        snippet = get_file_snippet(file_path)
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
        except OSError as e:
            failed.append((file_path.name, str(e)))
            continue

        state = {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_mb": f"{file_size_mb:.2f}MB",
            "content_preview": snippet,
        }

        try:
            res = router.predict(state, questions)
            answer = res["answers"]["category"]
            chosen_key = answer["choice"]
            conf = float(answer.get("confidence") or 0.0)
        except Exception as e:
            failed.append((file_path.name, f"分类失败: {e}"))
            tqdm.write(f"⚠ 分类失败 {file_path.name}: {e}")
            continue

        if chosen_key not in CATEGORIES:
            failed.append((file_path.name, f"未知分类: {chosen_key}"))
            tqdm.write(f"⚠ 未知分类 {file_path.name}: {chosen_key}")
            continue

        target_folder_name = CATEGORIES[chosen_key]["folder"]
        dest_dir = TARGET_BASE_DIR / target_folder_name
        dest_path = unique_dest(dest_dir, file_path)

        # 嵌套文件显示相对来源，方便对照
        try:
            rel_src = str(file_path.relative_to(downloads_dir))
        except ValueError:
            rel_src = file_path.name
        short_name = (
            rel_src[:34] + ".."
            if len(rel_src) > 36
            else rel_src
        )
        rel_dest = f"{target_folder_name}/{dest_path.name}"

        if conf < args.min_confidence:
            skipped.append((file_path.name, conf, target_folder_name))
            tqdm.write(
                f"{short_name:<36} {target_folder_name:<16} {conf:<8.2f}"
                f" ⚠ 置信度低于 {args.min_confidence}，跳过  [{rel_dest}]"
            )
            continue

        tqdm.write(
            f"{short_name:<36} {target_folder_name:<16} {conf:<8.2f} {rel_dest}"
        )
        moves.append((file_path, dest_path, dest_dir))

    bar.close()
    print("=" * 90, flush=True)

    if not args.apply:
        print("\n💡 当前为【预览演练模式】，文件未发生移动。", flush=True)
        if moves:
            print("👉 如果确认分类满意，请运行命令执行真实归档：", flush=True)
            print("   python download_organizer.py --apply", flush=True)
        if skipped:
            lowest = min(c for _, c, _ in skipped)
            # 建议阈值 = 略低于被跳过的最低分，保证这批都能过
            suggested = max(0.0, round(lowest - 0.01, 2))
            print(
                f"⚠ {len(skipped)} 个文件置信度低于当前阈值 {args.min_confidence} 已跳过"
                f"（其中最低 {lowest:.2f}）。",
                flush=True,
            )
            print(
                "👉 要把它们也纳入预览/归档，把阈值调到 0（全收）或调到 "
                f"{suggested} 以下，例如：",
                flush=True,
            )
            print("   python download_organizer.py --min-confidence 0", flush=True)
            if suggested > 0:
                print(
                    f"   python download_organizer.py --min-confidence {suggested}",
                    flush=True,
                )
            print(
                "   调完再加 --apply 即真实移动。",
                flush=True,
            )
        if failed:
            print(f"⚠ {len(failed)} 个文件分类失败：", flush=True)
            for name, err in failed:
                print(f"   - {name}: {err}", flush=True)
        print(flush=True)
        if failed and not moves:
            sys.exit(1)
        return

    print("\n🚀 正在执行物理归档移动...", flush=True)
    moved = 0
    move_failed = []
    move_bar = tqdm(
        moves,
        desc="移动归档",
        unit="file",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for src, dest, folder in move_bar:
        move_bar.set_postfix_str(src.name[:32], refresh=False)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            dest = unique_dest(folder, src)
            shutil.move(str(src), str(dest))
            moved += 1
        except Exception as e:
            move_failed.append((src.name, str(e)))
            tqdm.write(f"⚠ 移动失败 {src.name}: {e}")
    move_bar.close()

    print(
        f"✅ 归档完成：成功移动 {moved} 个文件到 {TARGET_BASE_DIR}/ 下对应分类目录。",
        flush=True,
    )
    if skipped:
        lowest = min(c for _, c, _ in skipped)
        print(
            f"⚠ 跳过低置信度 {len(skipped)} 个（阈值 {args.min_confidence}，"
            f"最低 {lowest:.2f}）。要一并移动可先 "
            f"--min-confidence 0 预览确认后再 --apply。",
            flush=True,
        )
    if failed:
        print(f"⚠ 分类失败 {len(failed)} 个：", flush=True)
        for name, err in failed:
            print(f"   - {name}: {err}", flush=True)
    if move_failed:
        print(f"⚠ 移动失败 {len(move_failed)} 个：", flush=True)
        for name, err in move_failed:
            print(f"   - {name}: {err}", flush=True)
        sys.exit(1)
    print(flush=True)


if __name__ == "__main__":
    main()
