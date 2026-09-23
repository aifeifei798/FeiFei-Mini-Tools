from collections import deque
import gc
import html
import os
import queue
import sys
import time
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)
from qwen_asr import Qwen3ASRModel
import sounddevice as sd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# 屏蔽无关警告日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.set_verbosity_error()

# 麦克风设备号：环境变量 LIVE_SUBTITLES_MIC 可覆盖，缺省用系统默认输入设备
MIC_DEVICE_ENV = "LIVE_SUBTITLES_MIC"


def _asr_text(res) -> str:
    if isinstance(res, list):
        if not res:
            return ""
        return (getattr(res[0], "text", "") or "").strip()
    return (getattr(res, "text", "") or "").strip()


def _translate(mt_tokenizer, mt_model, text, device="cuda:0") -> str:
    if not text.strip() or len(text.strip()) < 2:
        return ""
    prompt = f"将以下内容翻译成中文（如果是中文则翻译成英文）：\n{text}"
    messages = [{"role": "user", "content": prompt}]
    input_text = mt_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = mt_tokenizer(input_text, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    pad_id = mt_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = mt_tokenizer.eos_token_id
    with torch.no_grad():
        outputs = mt_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=pad_id,
        )
    return mt_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def _blend_hex(fg, alpha, bg=(15, 15, 15)) -> str:
    r = round(bg[0] + (fg[0] - bg[0]) * alpha)
    g = round(bg[1] + (fg[1] - bg[1]) * alpha)
    b = round(bg[2] + (fg[2] - bg[2]) * alpha)
    return f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"


# ================= 1. 5090 后台工作线程 =================
class SubtitleWorker(QThread):
    sig_status = pyqtSignal(str)
    sig_error = pyqtSignal(str)
    sig_interim = pyqtSignal(str)
    sig_final = pyqtSignal(str, str)

    VOL_THRESHOLD = 0.015

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        asr_model = None
        mt_model = None
        mt_tokenizer = None
        try:
            print(">>> 5090 正在载入模型...", flush=True)
            self.sig_status.emit("正在载入 ASR 模型...")
            asr_model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B",
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_new_tokens=128,
            )

            self.sig_status.emit("正在载入翻译模型...")
            if not self.is_running:
                return
            mt_model_id = "tencent/Hy-MT2-1.8B"
            mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_id)
            if not self.is_running:
                return
            mt_model = AutoModelForCausalLM.from_pretrained(
                mt_model_id, dtype=torch.bfloat16, device_map="cuda:0"
            )

            SAMPLE_RATE = 16000
            CHUNK_SIZE = int(SAMPLE_RATE * 0.2)
            audio_q = queue.Queue(maxsize=25)

            def audio_cb(indata, frames, time_info, status):
                if status:
                    print(f">>> 音频状态: {status}", flush=True)
                if not self.is_running:
                    return
                data = indata.copy()
                try:
                    audio_q.put_nowait(data)
                except queue.Full:
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        audio_q.put_nowait(data)
                    except queue.Full:
                        pass

            mic_env = os.environ.get(MIC_DEVICE_ENV, "").strip()
            mic_device = int(mic_env) if mic_env else None
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                device=mic_device,
                callback=audio_cb,
            )

            buffer = []
            silence_chunks = 0
            last_asr_time = time.time()
            cached_txt = ""
            cache_valid = False

            with stream:
                print(">>> 悬浮窗字幕引擎已完全就绪！", flush=True)
                self.sig_status.emit("引擎已就绪，开始聆听...")
                while self.is_running:
                    try:
                        chunk = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if not self.is_running:
                        break

                    vol = np.sqrt(np.mean(chunk**2))
                    if vol > self.VOL_THRESHOLD:
                        if cache_valid:
                            cache_valid = False
                        buffer.append(chunk)
                        silence_chunks = 0
                        if (
                            self.is_running
                            and time.time() - last_asr_time > 0.5
                            and len(buffer) >= 4
                        ):
                            try:
                                audio_data = np.concatenate(buffer).flatten()
                                res = asr_model.transcribe((audio_data, SAMPLE_RATE))
                                txt = _asr_text(res)
                                if txt:
                                    cached_txt = txt
                                    cache_valid = True
                                    self.sig_interim.emit(txt)
                            except Exception as e:
                                print(f">>> 中间识别失败: {e}", flush=True)
                            last_asr_time = time.time()
                        force_end = len(buffer) >= 25
                    else:
                        force_end = False
                        if len(buffer) > 0:
                            silence_chunks += 1
                            buffer.append(chunk)

                    if (
                        self.is_running
                        and (silence_chunks >= 2 and len(buffer) >= 4 or force_end)
                    ):
                        audio_data = np.concatenate(buffer).flatten()
                        buffer = []
                        silence_chunks = 0
                        if cache_valid and cached_txt:
                            final_src = cached_txt
                        else:
                            try:
                                res = asr_model.transcribe(
                                    (audio_data, SAMPLE_RATE)
                                )
                                final_src = _asr_text(res)
                            except Exception as e:
                                print(f">>> 最终识别失败: {e}", flush=True)
                                final_src = ""
                        cache_valid = False
                        if self.is_running and final_src and len(final_src) > 1:
                            try:
                                final_dst = _translate(
                                    mt_tokenizer, mt_model, final_src
                                )
                            except Exception as e:
                                print(f">>> 翻译失败: {e}", flush=True)
                                final_dst = ""
                            self.sig_final.emit(final_src, final_dst)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.sig_error.emit(f"引擎启动/运行失败: {e}")
        finally:
            asr_model = None
            mt_model = None
            mt_tokenizer = None
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f">>> 显存清理失败: {e}", flush=True)
            print(">>> 模型已卸载，显存已清理", flush=True)

    def request_stop(self):
        """仅置停止标志，绝不阻塞调用线程（必须在 UI 线程调用）。"""
        self.is_running = False

    def stop(self, timeout_ms=8000) -> bool:
        self.is_running = False
        return self.wait(timeout_ms)


# ================= 2. 支持缩放、挪动、退出的桌面 UI =================
class SubtitleWindow(QWidget):

    def __init__(self):
        super().__init__()
        # 窗口特性：无边框、永远置顶、透明底层
        # 注意：不能加 Qt.WindowType.SubWindow / Tool —— 这类"瞬态窗口"关闭后
        # 不会触发 lastWindowClosed，app.exec() 永不返回，点退出会假死。
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 窗口尺寸策略：允许自由缩放
        self.setMinimumSize(280, 100)
        self.resize(1000, 260)

        self.font_scale = 0
        self.history_out = deque(maxlen=5)
        self.drag_position = None

        # 主垂直布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(6)

        # ---------------- 顶部控制条 (拖拽条 + 按钮，默认隐藏，悬停显示) ----------------
        self.top_bar_widget = QWidget()
        top_bar = QHBoxLayout(self.top_bar_widget)
        top_bar.setContentsMargins(4, 0, 4, 0)

        self.title_label = QLabel("  5090 实时同传 (按住任意处可拖拽挪动)")
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: rgba(255, 255, 255, 140);")
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()

        btn_font_minus = QPushButton("A-")
        btn_font_minus.setFixedSize(28, 22)
        btn_font_minus.setStyleSheet(self._btn_style("#444"))
        btn_font_minus.setToolTip("缩小字体")
        btn_font_minus.clicked.connect(lambda: self.change_font_size(-1))
        top_bar.addWidget(btn_font_minus)

        btn_font_plus = QPushButton("A+")
        btn_font_plus.setFixedSize(28, 22)
        btn_font_plus.setStyleSheet(self._btn_style("#444"))
        btn_font_plus.setToolTip("放大字体")
        btn_font_plus.clicked.connect(lambda: self.change_font_size(1))
        top_bar.addWidget(btn_font_plus)

        btn_clear = QPushButton("清空")
        btn_clear.setFixedSize(40, 22)
        btn_clear.setStyleSheet(self._btn_style("#555"))
        btn_clear.setToolTip("清空字幕历史")
        btn_clear.clicked.connect(self.clear_history)
        top_bar.addWidget(btn_clear)

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(28, 22)
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(230, 50, 50, 200);
                color: white;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background: rgba(255, 30, 30, 250);
            }
        """)
        btn_close.setToolTip("退出同传软件 (或按 Esc)")
        btn_close.clicked.connect(self.close)
        top_bar.addWidget(btn_close)

        main_layout.addWidget(self.top_bar_widget)
        self.top_bar_widget.hide()

        # ---------------- 译文 5 行历史大框 ----------------
        self.out_box = QLabel()
        self.out_box.setStyleSheet("""
            background-color: rgba(15, 15, 15, 220);
            border-radius: 10px;
            padding: 10px 16px;
            border: 1px solid rgba(255, 255, 255, 30);
        """)
        self.out_box.setWordWrap(True)
        self.out_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self.out_box, stretch=1)

        # ---------------- 底部栏 (实时原文 + 缩放手柄) ----------------
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 0, 0, 0)

        self.live_label = QLabel("Waiting for speech...")
        self.live_label.setFont(QFont("Arial", 11, QFont.Weight.Medium))
        self.live_label.setStyleSheet("""
            color: #A0C4FF;
            background-color: rgba(20, 20, 20, 200);
            border-radius: 6px;
            padding: 4px 10px;
        """)
        self.live_label.setWordWrap(True)
        bottom_bar.addWidget(self.live_label, stretch=1)

        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("""
            QSizeGrip {
                width: 18px;
                height: 18px;
                background-color: rgba(255, 255, 255, 40);
                border-radius: 3px;
                margin-left: 6px;
            }
            QSizeGrip:hover {
                background-color: rgba(255, 215, 0, 180);
            }
        """)
        bottom_bar.addWidget(
            self.size_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
        )

        main_layout.addLayout(bottom_bar)
        self.setLayout(main_layout)

        self._place_on_screen()
        self.render_history(">>> 正在启动引擎，加载模型中，请稍候...")

        QShortcut(QKeySequence("Esc"), self, self.close)

        self.worker = SubtitleWorker()
        self.worker.sig_status.connect(self.show_status)
        self.worker.sig_error.connect(self.show_error)
        self.worker.sig_interim.connect(self.update_interim)
        self.worker.sig_final.connect(self.update_final)
        self.worker.start()

    def _place_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.move(400, 720)
            return
        area = screen.availableGeometry()
        x = area.x() + max(0, (area.width() - self.width()) // 2)
        y = area.y() + int(area.height() * 0.65)
        x = max(area.x(), min(x, area.x() + area.width() - self.width()))
        y = max(area.y(), min(y, area.y() + area.height() - self.height()))
        self.move(x, y)

    def _btn_style(self, bg_color):
        return f"""
            QPushButton {{
                background: {bg_color};
                color: #DDD;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background: #777;
                color: white;
            }}
        """

    def change_font_size(self, delta):
        self.font_scale = max(-4, min(10, self.font_scale + delta))
        self.render_history()

    def clear_history(self):
        self.history_out.clear()
        self.render_history(">>> 历史已清空，静候音频...")

    def show_status(self, msg):
        print(f">>> {msg}", flush=True)
        self.render_history(f">>> {msg}")

    def show_error(self, msg):
        print(f">>> 错误: {msg}", flush=True)
        self.render_history(f">>> 错误: {msg}")

    def render_history(self, new_text=None):
        if new_text:
            self.history_out.append(new_text)

        html_snippets = []
        total = len(self.history_out)
        base_size = 17 + self.font_scale
        hist_size = max(11, base_size - 4)

        for idx, text in enumerate(self.history_out):
            safe_text = html.escape(text)
            if idx == total - 1:
                snippet = (
                    f'<div style="color: #FFD700; font-size: {base_size}px;'
                    ' font-weight: bold; margin-top: 4px; line-height:'
                    f' 140%;">{safe_text}</div>'
                )
            else:
                opacity = 0.35 + (idx / total) * 0.35
                hist_color = _blend_hex((230, 230, 230), opacity)
                snippet = (
                    f'<div style="color: {hist_color};'
                    f' font-size: {hist_size}px; margin-top: 2px; line-height:'
                    f' 130%;">{safe_text}</div>'
                )
            html_snippets.append(snippet)

        self.out_box.setText("".join(html_snippets))

    def update_interim(self, text):
        self.live_label.setText(f"🗣️  {text}")

    def update_final(self, src_text, trans_text):
        self.live_label.setText(f"🗣️  {src_text}")
        self.render_history(trans_text or src_text)

    # 鼠标移入窗口 -> 显示顶部控制条；移出 -> 隐藏
    def enterEvent(self, event):
        self.top_bar_widget.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.top_bar_widget.hide()
        super().leaveEvent(event)

    # 鼠标左键点击任意非按钮区域 -> 自由挪动位置
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self.drag_position is not None
        ):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    # 关闭窗口：只置停止位，不阻塞 UI；线程回收放到事件循环结束后
    def closeEvent(self, event):
        print(">>> 正在退出同传软件...", flush=True)
        self.worker.request_stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    win = SubtitleWindow()
    win.show()
    code = app.exec()
    # 事件循环已退出，再同步回收工作线程（加载/推理中最多等 timeout）
    if not win.worker.stop(timeout_ms=8000):
        print(">>> 工作线程未能按时退出，强制结束进程", flush=True)
        os._exit(code if isinstance(code, int) else 0)
    print(">>> 已安全退出", flush=True)
    sys.exit(code)
