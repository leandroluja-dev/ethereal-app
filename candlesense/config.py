"""Configuração: valores padrão no código, sobrescritos por config.yaml (opcional).

O app funciona sem nenhum arquivo de configuração. O config.yaml só precisa conter
as chaves que você quer mudar.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

log = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    fps: float = 8.0
    # Expressões regulares (sem diferenciar maiúsculas) para achar a janela pelo título.
    # Cobre o app desktop ("IQ Option") e abas do navegador ("... IQ Option - Google Chrome").
    title_patterns: list[str] = field(default_factory=lambda: [r"iq\s*option", r"iqoption"])
    exclude_title_patterns: list[str] = field(default_factory=lambda: [r"candlesense"])
    panel_width: int = 360
    # Esconde o painel das capturas de tela (Windows 10 2004+), para ele poder cobrir a IQ Option.
    exclude_panel_from_capture: bool = True
    # Intervalo para revalidar a janela (posição, tamanho, monitor).
    window_check_interval_s: float = 1.0


@dataclass
class VisionConfig:
    min_body_width_px: int = 3
    min_pitch_px: int = 5
    min_candles: int = 15
    chroma_min: int = 50
    val_min: int = 60
    hue_tolerance: int = 12
    bg_tolerance: float = 28.0
    color_refresh_s: float = 30.0
    axis_change_threshold: float = 3.0
    hud_ocr_interval_s: float = 2.0
    layout_retry_s: float = 3.0
    min_confidence: float = 0.6
    # "auto": usa o relógio (velas alinhadas ao minuto) e confere com o OCR; "ocr" ou "clock" forçam.
    timer_source: str = "auto"
    clock_offset_s: float = 0.0
    # Se o eixo de tempo não puder ser lido, você pode fixar o timeframe aqui (1 ou 5). null = detectar.
    assume_timeframe_min: int | None = None
    # Retângulo manual da área das velas [x, y, w, h] relativo à janela (plano B). null = automático.
    manual_plot_rect: list[int] | None = None


@dataclass
class OcrConfig:
    backend: str = "auto"  # auto | rapidocr | tesseract
    tesseract_cmd: str | None = None
    min_text_height_px: int = 24


@dataclass
class ReplayConfig:
    dir: str = "recordings"
    record_fps: float = 2.0
    image_format: str = "png"


@dataclass
class UiConfig:
    always_on_top: bool = True
    debug_on_start: bool = False
    opacity: float = 0.97


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "logs"


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    source_path: str | None = None


def app_dir() -> Path:
    """Pasta do app: ao lado do .exe (PyInstaller) ou a pasta atual."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _coerce(value: Any, typ: Any, path: str) -> Any:
    if dataclasses.is_dataclass(typ) and isinstance(typ, type):
        if not isinstance(value, dict):
            log.warning("config: %s deveria ser um bloco; ignorado", path)
            return typ()
        return _from_dict(typ, value, path)
    if value is None:
        return None
    try:
        if typ is float:
            return float(value)
        if typ is int:
            return int(value)
        if typ is bool:
            return bool(value)
        if typ is str:
            return str(value)
    except (TypeError, ValueError):
        log.warning("config: valor inválido em %s: %r", path, value)
        raise
    return value


def _from_dict(cls: type, data: dict[str, Any], path: str = "") -> Any:
    hints = get_type_hints(cls)
    obj = cls()
    for key, value in data.items():
        if key not in hints:
            log.warning("config: chave desconhecida %s%s (ignorada)", path, key)
            continue
        try:
            setattr(obj, key, _coerce(value, hints[key], f"{path}{key}."))
        except (TypeError, ValueError):
            pass
    return obj


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


USER_OVERRIDES_FILE = "candlesense_user.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Carrega config.yaml (se existir) + candlesense_user.yaml (ajustes salvos pelo app)."""
    candidates = [Path(path)] if path else [app_dir() / "config.yaml"]
    merged: dict[str, Any] = {}
    used: str | None = None
    for p in candidates:
        if p.is_file():
            try:
                merged = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                used = str(p)
            except Exception:  # noqa: BLE001 - config ruim nunca derruba o app
                log.exception("config: falha ao ler %s; usando padrões", p)
            break
    user = app_dir() / USER_OVERRIDES_FILE
    if user.is_file():
        try:
            merged = _deep_merge(merged, yaml.safe_load(user.read_text(encoding="utf-8")) or {})
        except Exception:  # noqa: BLE001
            log.exception("config: falha ao ler %s", user)
    if not isinstance(merged, dict):
        merged = {}
    cfg: Config = _from_dict(Config, merged)
    cfg.source_path = used
    return cfg


def save_user_override(section: str, key: str, value: Any) -> None:
    """Grava um ajuste feito pela interface (ex.: área manual) em candlesense_user.yaml."""
    path = app_dir() / USER_OVERRIDES_FILE
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            data = {}
    data.setdefault(section, {})[key] = value
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
