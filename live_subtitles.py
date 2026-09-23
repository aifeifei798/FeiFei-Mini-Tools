from collections import deque
import gc
import html
import os
import queue
import sys
import time
import numpy as np
from PyQt6.QtCore import QLocale, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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

# 字号（px）：译文历史最新行 = FONT_BASE_PX + font_scale
# 下限照顾 1920x1080 全屏，上限给 4K 全屏放大
FONT_BASE_PX = 17
FONT_PX_MIN = 9
FONT_PX_MAX = 48

# 语言表：(英语名/键, 中文名, Qwen3-ASR 支持, Hy-MT2 支持, 系统 locale 语言码)
# 源语言下拉只给 ASR 支持的，译文语言下拉只给 Hy-MT2 支持的
LANG_ROWS = [
    ("Chinese", "中文", True, True, ("zh",)),
    ("Traditional Chinese", "繁体中文", False, True, ()),
    ("English", "英语", True, True, ("en",)),
    ("Cantonese", "粤语", True, True, ()),
    ("Japanese", "日语", True, True, ("ja",)),
    ("Korean", "韩语", True, True, ("ko",)),
    ("French", "法语", True, True, ("fr",)),
    ("German", "德语", True, True, ("de",)),
    ("Spanish", "西班牙语", True, True, ("es",)),
    ("Portuguese", "葡萄牙语", True, True, ("pt",)),
    ("Italian", "意大利语", True, True, ("it",)),
    ("Russian", "俄语", True, True, ("ru",)),
    ("Arabic", "阿拉伯语", True, True, ("ar",)),
    ("Thai", "泰语", True, True, ("th",)),
    ("Vietnamese", "越南语", True, True, ("vi",)),
    ("Malay", "马来语", True, True, ("ms",)),
    ("Indonesian", "印尼语", True, True, ("id",)),
    ("Filipino", "菲律宾语", True, True, ("tl", "fil")),
    ("Hindi", "印地语", True, True, ("hi",)),
    ("Turkish", "土耳其语", True, True, ("tr",)),
    ("Dutch", "荷兰语", True, True, ("nl",)),
    ("Polish", "波兰语", True, True, ("pl",)),
    ("Czech", "捷克语", True, True, ("cs",)),
    ("Persian", "波斯语", True, True, ("fa",)),
    ("Ukrainian", "乌克兰语", False, True, ("uk",)),
    ("Swedish", "瑞典语", True, False, ()),
    ("Danish", "丹麦语", True, False, ()),
    ("Finnish", "芬兰语", True, False, ()),
    ("Greek", "希腊语", True, False, ()),
    ("Romanian", "罗马尼亚语", True, False, ()),
    ("Hungarian", "匈牙利语", True, False, ()),
    ("Macedonian", "马其顿语", True, False, ()),
    ("Khmer", "高棉语", False, True, ("km",)),
    ("Burmese", "缅甸语", False, True, ("my",)),
    ("Gujarati", "古吉拉特语", False, True, ("gu",)),
    ("Urdu", "乌尔都语", False, True, ("ur",)),
    ("Telugu", "泰卢固语", False, True, ("te",)),
    ("Marathi", "马拉地语", False, True, ("mr",)),
    ("Hebrew", "希伯来语", False, True, ("he",)),
    ("Bengali", "孟加拉语", False, True, ("bn",)),
    ("Tamil", "泰米尔语", False, True, ("ta",)),
    ("Tibetan", "藏语", False, True, ()),
    ("Kazakh", "哈萨克语", False, True, ("kk",)),
    ("Mongolian", "蒙古语", False, True, ("mn",)),
    ("Uyghur", "维吾尔语", False, True, ("ug",)),
]
MT_ZH_BY_KEY = {r[0]: r[1] for r in LANG_ROWS if r[3]}
LOCALE_TO_KEY = {lc: r[0] for r in LANG_ROWS for lc in r[4]}


def _system_lang_key() -> str:
    """译文语言选 auto 时，解析系统语言为语言键；不支持则回退英语。"""
    try:
        name = QLocale.system().name() or ""  # 例如 zh_CN
    except Exception:
        name = ""
    if not name:
        name = os.environ.get("LANG", "")
    name = name.split(".")[0].replace("-", "_")
    lang, _, terr = name.partition("_")
    lang = lang.lower()
    if lang == "zh":
        if terr.upper() in ("TW", "HK", "MO"):
            return "Traditional Chinese"
        return "Chinese"
    return LOCALE_TO_KEY.get(lang, "English")


def _asr_parts(res) -> tuple:
    """返回 (语言, 文本)；语言为 ASR 检测到的英语名，识别失败为空。"""
    if isinstance(res, list):
        if not res:
            return "", ""
        res = res[0]
    lang = (getattr(res, "language", "") or "").strip()
    text = (getattr(res, "text", "") or "").strip()
    return lang, text


def _translate(mt_tokenizer, mt_model, text, target_zh, device="cuda:0") -> str:
    if not text.strip() or len(text.strip()) < 2:
        return ""
    prompt = (
        f"将以下文本翻译为{target_zh}，"
        f"注意只需要输出翻译后的结果，不要额外解释：\n{text}"
    )
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
        # 源语言：ASR 英语名（如 "Chinese"），None = auto 模型自己识别
        self.src_lang = None
        # 译文语言键（如 "Chinese"），None = auto 取系统语言
        self.tgt_lang = None

    def run(self):
        asr_model = None
        mt_model = None
        mt_tokenizer = None
        try:
            sys_lang_key = _system_lang_key()
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
            cached_lang = ""
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
                                res = asr_model.transcribe(
                                    (audio_data, SAMPLE_RATE),
                                    language=self.src_lang,
                                )
                                lang, txt = _asr_parts(res)
                                if txt:
                                    cached_lang = lang
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
                            final_lang = cached_lang
                            final_src = cached_txt
                        else:
                            try:
                                res = asr_model.transcribe(
                                    (audio_data, SAMPLE_RATE),
                                    language=self.src_lang,
                                )
                                final_lang, final_src = _asr_parts(res)
                            except Exception as e:
                                print(f">>> 最终识别失败: {e}", flush=True)
                                final_lang, final_src = "", ""
                        cache_valid = False
                        if self.is_running and final_src and len(final_src) > 1:
                            tgt_key = self.tgt_lang or sys_lang_key
                            tgt_zh = MT_ZH_BY_KEY.get(tgt_key, "中文")
                            if final_lang and final_lang == tgt_key:
                                final_dst = final_src
                            else:
                                try:
                                    final_dst = _translate(
                                        mt_tokenizer, mt_model, final_src, tgt_zh
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

        # 窗口尺寸策略：允许自由缩放；默认尺寸在 _place_on_screen 里按屏幕算
        self.setMinimumSize(400, 110)

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

        self.title_label = QLabel("  5090 实时同传")
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: rgba(255, 255, 255, 140);")
        self.title_label.setToolTip("按住任意处可拖拽挪动；右下角手柄缩放窗口")
        top_bar.addWidget(self.title_label)

        lang_label_style = "color: rgba(255, 255, 255, 140); font-size: 11px;"

        lbl_src = QLabel("识别:")
        lbl_src.setStyleSheet(lang_label_style)
        top_bar.addWidget(lbl_src)

        self.combo_src = QComboBox()
        self.combo_src.addItem("auto 自动识别", None)
        for key, zh, asr_ok, _mt_ok, _locales in LANG_ROWS:
            if asr_ok:
                self.combo_src.addItem(zh, key)
        self.combo_src.setFixedHeight(22)
        self.combo_src.setMinimumWidth(88)
        self.combo_src.setStyleSheet(self._combo_style())
        self.combo_src.setToolTip("源语言（识别语种）：auto = 模型自己识别")
        top_bar.addWidget(self.combo_src)

        lbl_tgt = QLabel("译文:")
        lbl_tgt.setStyleSheet(lang_label_style)
        top_bar.addWidget(lbl_tgt)

        self.combo_tgt = QComboBox()
        self.combo_tgt.addItem("auto 系统语言", None)
        for key, zh, _asr_ok, mt_ok, _locales in LANG_ROWS:
            if mt_ok:
                self.combo_tgt.addItem(zh, key)
        self.combo_tgt.setFixedHeight(22)
        self.combo_tgt.setMinimumWidth(96)
        self.combo_tgt.setStyleSheet(self._combo_style())
        self.combo_tgt.setToolTip(
            "译文语言：auto = 系统语言（当前解析为 "
            f"{MT_ZH_BY_KEY.get(_system_lang_key(), '英语')}）"
        )
        top_bar.addWidget(self.combo_tgt)

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

        # 字号直选：与 A-/A+ 双向同步
        self.combo_font = QComboBox()
        for px in range(FONT_PX_MIN, FONT_PX_MAX + 1):
            self.combo_font.addItem(f"{px}px", px)
        self.combo_font.setFixedHeight(22)
        self.combo_font.setMinimumWidth(58)
        self.combo_font.setStyleSheet(self._combo_style())
        self.combo_font.setToolTip(
            f"字号选择（{FONT_PX_MIN}~{FONT_PX_MAX}px）："
            "1080p 建议 9~16，4K 全屏可放大到 32+"
        )
        init_font_idx = self.combo_font.findData(FONT_BASE_PX)
        if init_font_idx >= 0:
            self.combo_font.setCurrentIndex(init_font_idx)
        top_bar.addWidget(self.combo_font)

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
        self._apply_font()
        self.render_history(">>> 正在启动引擎，加载模型中，请稍候...")

        QShortcut(QKeySequence("Esc"), self, self.close)

        self.worker = SubtitleWorker()
        self.worker.sig_status.connect(self.show_status)
        self.worker.sig_error.connect(self.show_error)
        self.worker.sig_interim.connect(self.update_interim)
        self.worker.sig_final.connect(self.update_final)
        self.combo_src.currentIndexChanged.connect(self._on_src_lang_changed)
        self.combo_tgt.currentIndexChanged.connect(self._on_tgt_lang_changed)
        self.combo_font.currentIndexChanged.connect(self._on_font_changed)
        self.worker.start()

    def _place_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(800, 220)
            self.move(400, 720)
            return
        area = screen.availableGeometry()
        # 默认尺寸按屏幕比例：1080p 收敛些，4K 不至于太迷你
        w = max(480, min(1400, int(area.width() * 0.52)))
        h = max(160, min(420, int(area.height() * 0.24)))
        self.resize(w, h)
        x = area.x() + max(0, (area.width() - w) // 2)
        y = area.y() + int(area.height() * 0.65)
        x = max(area.x(), min(x, area.x() + area.width() - w))
        y = max(area.y(), min(y, area.y() + area.height() - h))
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

    def _combo_style(self):
        return """
            QComboBox {
                background: rgba(40, 40, 40, 200);
                color: #DDD;
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 4px;
                padding: 1px 6px;
                font-size: 11px;
            }
            QComboBox:hover, QComboBox:focus {
                border-color: rgba(255, 255, 255, 120);
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                color: #EEE;
                border: 1px solid #555;
                selection-background-color: #555;
                selection-color: white;
            }
        """

    def _on_src_lang_changed(self, index):
        key = self.combo_src.itemData(index)
        self.worker.src_lang = key
        label = key or "auto（模型自动识别）"
        print(f">>> 识别语言已切换: {label}", flush=True)

    def _on_tgt_lang_changed(self, index):
        key = self.combo_tgt.itemData(index)
        self.worker.tgt_lang = key
        if key:
            label = key
        else:
            sys_key = _system_lang_key()
            label = f"auto（系统语言 → {MT_ZH_BY_KEY.get(sys_key, sys_key)}）"
        print(f">>> 译文语言已切换: {label}", flush=True)

    def _font_px(self) -> int:
        return FONT_BASE_PX + self.font_scale

    def _set_font_px(self, px: int):
        px = max(FONT_PX_MIN, min(FONT_PX_MAX, px))
        self.font_scale = px - FONT_BASE_PX
        idx = self.combo_font.findData(px)
        if idx >= 0 and self.combo_font.currentIndex() != idx:
            self.combo_font.blockSignals(True)
            self.combo_font.setCurrentIndex(idx)
            self.combo_font.blockSignals(False)
        self._apply_font()

    def _on_font_changed(self, index):
        px = self.combo_font.itemData(index)
        if px is not None:
            self.font_scale = int(px) - FONT_BASE_PX
            self._apply_font()

    def _apply_font(self):
        self.render_history()
        # 底部实时原文跟着缩放，略小一号
        if hasattr(self, "live_label"):
            f = self.live_label.font()
            f.setPixelSize(max(FONT_PX_MIN, self._font_px() - 4))
            self.live_label.setFont(f)

    def change_font_size(self, delta):
        self._set_font_px(self._font_px() + delta)

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
        base_size = max(FONT_PX_MIN, FONT_BASE_PX + self.font_scale)
        hist_size = max(FONT_PX_MIN, base_size - 4)

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
        # 语言下拉展开时鼠标会移出主窗口，此时不能收起顶栏，否则选项点不到
        if not self._lang_combo_popup_open():
            self.top_bar_widget.hide()
        super().leaveEvent(event)

    def _lang_combo_popup_open(self) -> bool:
        for combo in (self.combo_src, self.combo_tgt, self.combo_font):
            if combo.view().window().isVisible():
                return True
        return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 窄窗口先藏标题，把空间让给语言/字号控件
        if hasattr(self, "title_label"):
            self.title_label.setVisible(self.width() >= 640)

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
