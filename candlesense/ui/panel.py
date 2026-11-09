"""Painel compacto, sempre no topo, ao lado da IQ Option.

Fase 1: mostra só o que a visão leu (ativo, payout, timeframe, cronômetro, preço, velas e confiança).
A caixa de direção fica cinza: sinais e probabilidades chegam na fase 2, sempre calculados pelo motor.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..vision.analyzer import Reading, Status

STATUS_COLORS = {
    Status.OK: "#1f7a3a",
    Status.UNCERTAIN: "#9a6b00",
    Status.ZOOM: "#9a6b00",
    Status.DETECTING: "#3d4a63",
    Status.PAUSED: "#3d4a63",
}
DEFAULT_STATUS_COLOR = "#8a2b2b"

STYLE = """
QWidget { background: #161922; color: #e6e8ee; font-size: 13px; }
QLabel[class="k"] { color: #8b93a7; }
QPushButton { background: #252a37; border: 1px solid #343b4d; padding: 6px 8px; border-radius: 4px; }
QPushButton:hover { background: #2f3545; }
QPushButton:checked { background: #3b4a6b; }
QPushButton:disabled { color: #5d6478; }
QProgressBar { border: 1px solid #343b4d; border-radius: 3px; text-align: center; background: #1d212c; height: 14px; }
QProgressBar::chunk { background: #3d7bd9; }
"""


def _fmt_timer(s: float | None) -> str:
    if s is None:
        return "--:--"
    s = max(0, int(s))
    return f"{s // 60:02d}:{s % 60:02d}"


class Panel(QWidget):
    adjust_clicked = pyqtSignal()
    debug_toggled = pyqtSignal(bool)
    pause_toggled = pyqtSignal(bool)
    record_clicked = pyqtSignal()
    snapshot_clicked = pyqtSignal()
    relayout_clicked = pyqtSignal()

    def __init__(self, always_on_top: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("CandleSense")
        flags = Qt.WindowType.Window
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(300)

        self.status = QLabel("Iniciando…")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        f = QFont()
        f.setPointSize(12)
        f.setBold(True)
        self.status.setFont(f)

        self.direction = QLabel("SEM SINAL\nfase 1: somente leitura")
        self.direction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fd = QFont()
        fd.setPointSize(15)
        fd.setBold(True)
        self.direction.setFont(fd)
        self.direction.setStyleSheet("background:#3a3f4b; color:#c9ced8; border-radius:6px; padding:10px;")

        self.timer = QLabel("--:--")
        ft = QFont()
        ft.setPointSize(22)
        ft.setBold(True)
        self.timer.setFont(ft)
        self.timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_src = QLabel("")
        self.timer_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_src.setProperty("class", "k")

        grid = QGridLayout()
        grid.setVerticalSpacing(3)
        self.fields: dict[str, QLabel] = {}
        rows = [("ativo", "Ativo"), ("tf", "Timeframe"), ("payout", "Payout"), ("price", "Preço atual"),
                ("forming", "Vela atual"), ("last", "Última fechada"), ("ncandles", "Velas lidas"),
                ("window", "Janela"), ("latency", "Latência")]
        for i, (key, name) in enumerate(rows):
            k = QLabel(name)
            k.setProperty("class", "k")
            v = QLabel("—")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.setWordWrap(True)
            grid.addWidget(k, i, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(v, i, 1)
            self.fields[key] = v
        grid.setColumnStretch(1, 1)

        self.conf = QProgressBar()
        self.conf.setRange(0, 100)
        self.conf.setFormat("confiança da leitura %p%")
        self.conf_parts = QLabel("")
        self.conf_parts.setWordWrap(True)
        self.conf_parts.setStyleSheet("color:#8b93a7; font-size:11px;")
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color:#e0b252; font-size:12px;")
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#8b93a7; font-size:11px;")

        b_adjust = QPushButton("Ajustar janela")
        b_adjust.setToolTip("Reposiciona a IQ Option e coloca este painel ao lado (sem mouse/teclado).")
        b_adjust.clicked.connect(self.adjust_clicked.emit)
        self.b_debug = QPushButton("Debug")
        self.b_debug.setCheckable(True)
        self.b_debug.toggled.connect(self.debug_toggled.emit)
        b_score = QPushButton("Placar")
        b_score.setEnabled(False)
        b_score.setToolTip("Disponível na fase 3")
        self.b_pause = QPushButton("Pausar")
        self.b_pause.setCheckable(True)
        self.b_pause.toggled.connect(self._on_pause)
        self.b_record = QPushButton("Gravar sessão")
        self.b_record.setCheckable(True)
        self.b_record.clicked.connect(self.record_clicked.emit)
        b_snap = QPushButton("Salvar captura")
        b_snap.clicked.connect(self.snapshot_clicked.emit)
        b_relayout = QPushButton("Redetectar")
        b_relayout.setToolTip("Procura de novo o gráfico, o eixo e o payout na janela.")
        b_relayout.clicked.connect(self.relayout_clicked.emit)

        row1 = QHBoxLayout()
        for b in (b_adjust, self.b_debug, b_score, self.b_pause):
            row1.addWidget(b)
        row2 = QHBoxLayout()
        for b in (self.b_record, b_snap, b_relayout):
            row2.addWidget(b)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#2a2f3c;")

        lay = QVBoxLayout(self)
        lay.addWidget(self.status)
        lay.addWidget(self.direction)
        lay.addWidget(self.timer)
        lay.addWidget(self.timer_src)
        lay.addWidget(line)
        lay.addLayout(grid)
        lay.addWidget(self.conf)
        lay.addWidget(self.conf_parts)
        lay.addWidget(self.notes)
        lay.addStretch(1)
        lay.addWidget(self.info)
        lay.addLayout(row1)
        lay.addLayout(row2)
        self.resize(360, 640)

    def _on_pause(self, on: bool) -> None:
        self.b_pause.setText("Continuar" if on else "Pausar")
        self.pause_toggled.emit(on)

    def set_info(self, text: str) -> None:
        self.info.setText(text)

    def update_reading(self, r: Reading) -> None:
        color = STATUS_COLORS.get(r.status, DEFAULT_STATUS_COLOR)
        self.status.setText(r.message)
        self.status.setStyleSheet(f"background:{color}; color:white; border-radius:6px; padding:8px;")
        ok = r.status in (Status.OK, Status.UNCERTAIN, Status.ZOOM)
        m = r.axis.mapping if (r.axis is not None and r.axis.mapping is not None) else None

        def price(v: float | None) -> str:
            if v is None:
                return "—"
            return m.fmt(v) if m else f"{v:.5f}"

        self.timer.setText(_fmt_timer(r.timer_s) if ok else "--:--")
        self.timer_src.setText(f"cronômetro da vela ({r.timer_source})" if ok and r.timer_source else "")
        self.fields["ativo"].setText(r.asset or "—")
        tf = f"M{r.timeframe_min}" if r.timeframe_min else "—"
        self.fields["tf"].setText(tf + (f"  ({r.timeframe_source})" if r.timeframe_source else ""))
        self.fields["payout"].setText(f"{r.payout}%" if r.payout is not None else "—")
        self.fields["price"].setText(price(r.current_price) + (f"  ({r.current_price_source})"
                                                               if r.current_price_source else ""))
        f = r.forming
        self.fields["forming"].setText("—" if f is None else
                                       f"A {price(f.open)}  Máx {price(f.high)}  Mín {price(f.low)}")
        last = r.candles[-2] if len(r.candles) >= 2 else None
        self.fields["last"].setText("—" if last is None else
                                    f"{'▲' if last.bull else '▼'} A {price(last.open)} F {price(last.close)}\n"
                                    f"Máx {price(last.high)}  Mín {price(last.low)}")
        occ = sum(c.occluded for c in r.candles)
        self.fields["ncandles"].setText(f"{len(r.candles)}" + (f"  ({occ} cobertas)" if occ else "") if ok else "—")
        self.fields["window"].setText(r.window_title or "—")
        total = r.timings_ms.get("total")
        self.fields["latency"].setText(f"{total:.0f} ms" if total is not None else "—")
        self.conf.setValue(int(round(r.confidence * 100)))
        self.conf_parts.setText("  ".join(f"{k} {v:.0%}" for k, v in r.conf_parts.items()))
        notes = list(r.events) + list(r.notes)
        self.notes.setText("\n".join(notes[:4]))
