"""Motor por regras: junta as notas dos módulos numa probabilidade por expiração.

Honestidade antes de tudo:

* Sem leitura OK, não há sinal ("leitura incerta").
* A probabilidade sai de uma soma ponderada das notas e é presa num teto
  baixo (``max_edge``). As regras ainda não foram medidas contra o resultado
  real; isso é o placar da Fase 3 e a calibração da Fase 4. Até lá o motor
  prefere dizer NÃO ENTRAR.
* Só sinaliza quando a probabilidade passa do ponto de equilíbrio do payout,
  1 / (1 + payout), mais uma margem; só dentro do tempo de compra; e só
  quando vários módulos concordam.
* O app nunca opera: a decisão é só uma indicação na tela.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..models import ChartReading, fmt_duration
from ..signals.modules import MarketContext, ModuleResult, run_all

COMPRA = "COMPRA"
VENDA = "VENDA"
NAO_ENTRAR = "NÃO ENTRAR"
INCERTA = "LEITURA INCERTA"

HORIZONS = (1, 2, 3, 5)

# Peso de cada módulo por expiração (1, 2, 3 e 5 velas). A vela atual pesa
# muito no curtíssimo prazo e quase nada em 5 velas; a tendência, o contrário.
DEFAULT_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "tendencia": (0.25, 0.30, 0.35, 0.40),
    "sr": (0.25, 0.25, 0.25, 0.20),
    "price_action": (0.25, 0.20, 0.15, 0.10),
    "momentum": (0.25, 0.20, 0.15, 0.10),
    "volatilidade": (0.10, 0.10, 0.10, 0.10),
    "micro": (0.30, 0.15, 0.08, 0.04),
}


@dataclass
class EngineConfig:
    horizons: tuple[int, ...] = HORIZONS
    weights: dict[str, tuple[float, ...]] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # Maior distância de 50% que as regras podem afirmar, por expiração.
    # Sem calibração, 60% é o máximo que o motor se permite dizer.
    max_edge: tuple[float, ...] = (0.10, 0.09, 0.08, 0.07)
    margin: float = 0.02  # acima do equilíbrio do payout
    # O começo da vela é o momento preferido; depois, exige mais.
    phase_margin: dict[str, float] = field(default_factory=lambda: {"início": 0.0, "meio": 0.01, "fim": 0.02})
    min_seconds_to_enter: int = 5
    min_agree: int = 3  # módulos com opinião na mesma direção
    agree_threshold: float = 0.15  # |nota x confiança| para contar como opinião
    max_strong_against: int = 1
    min_closed_candles: int = 20
    stable_readings: int = 2  # mesma decisão em N leituras seguidas antes de acender


@dataclass
class HorizonProb:
    candles: int
    p_up: float

    @property
    def direction(self) -> str:
        return COMPRA if self.p_up >= 0.5 else VENDA

    @property
    def prob(self) -> float:
        """Probabilidade da direção mais provável."""
        return max(self.p_up, 1.0 - self.p_up)


@dataclass
class Decision:
    action: str  # COMPRA | VENDA | NÃO ENTRAR | LEITURA INCERTA
    reasons: list[str] = field(default_factory=list)  # por que não entrar (ou o motivo da incerteza)
    horizons: list[HorizonProb] = field(default_factory=list)
    best: Optional[HorizonProb] = None
    breakeven: Optional[float] = None
    threshold: Optional[float] = None  # probabilidade mínima para sinalizar agora
    modules: list[ModuleResult] = field(default_factory=list)
    phase: Optional[str] = None  # início | meio | fim da vela
    seconds_to_enter: Optional[int] = None
    timeframe_s: Optional[int] = None
    target: Optional[float] = None
    exit_mode: str = ""
    expiry_hint: str = ""  # comparação com a expiração escolhida na IQ
    agree: int = 0
    decimals: int = 5

    @property
    def signal(self) -> bool:
        return self.action in (COMPRA, VENDA)

    def expiry_text(self, h: Optional[HorizonProb] = None) -> str:
        h = h or self.best
        if h is None:
            return ""
        n = f"{h.candles} vela" + ("s" if h.candles > 1 else "")
        if self.timeframe_s:
            return f"{n} ({fmt_duration(h.candles * self.timeframe_s)})"
        return n

    def summary(self) -> str:
        parts = [self.action]
        if self.best is not None:
            parts.append(f"{self.best.direction} {pct(self.best.prob)} em {self.expiry_text()}")
        if self.breakeven is not None:
            parts.append(f"equilíbrio {pct(self.breakeven)}")
        if self.seconds_to_enter is not None:
            parts.append(f"entrar em até {self.seconds_to_enter}s")
        if self.reasons:
            parts.append("; ".join(self.reasons))
        return " | ".join(parts)


def pct(p: float) -> str:
    """0.538 -> '53,8%'."""
    return f"{p * 100:.1f}%".replace(".", ",")


def breakeven(payout_pct: float) -> float:
    """Acerto mínimo para não perder dinheiro: 1 / (1 + payout). 86% -> 53,8%."""
    return 1.0 / (1.0 + payout_pct / 100.0)


def candle_phase(elapsed_frac: Optional[float]) -> Optional[str]:
    if elapsed_frac is None:
        return None
    if elapsed_frac < 1 / 3:
        return "início"
    if elapsed_frac < 2 / 3:
        return "meio"
    return "fim"


class RuleEngine:
    def __init__(self, cfg: Optional[EngineConfig] = None) -> None:
        self.cfg = cfg or EngineConfig()
        self._last_action: Optional[str] = None
        self._streak = 0

    def reset(self) -> None:
        self._last_action = None
        self._streak = 0

    # ------------------------------------------------------------------ API

    def evaluate(self, reading: ChartReading) -> Decision:
        d = self._evaluate(reading)
        return self._stabilize(d)

    def probabilities(self, mods: list[ModuleResult], quality: float = 1.0) -> list[HorizonProb]:
        """Probabilidade de alta por expiração, a partir das notas dos módulos."""
        cfg = self.cfg
        fator = 1.0
        for m in mods:
            if m.key == "volatilidade":
                fator = float(m.details.get("fator", 1.0))
        out = []
        for i, h in enumerate(cfg.horizons):
            num = den = 0.0
            for m in mods:
                w = cfg.weights.get(m.key, (0.0,) * len(cfg.horizons))[i]
                num += w * m.score * m.confidence
                den += w * m.confidence
            # Média ponderada pela confiança: módulo sem opinião pesa pouco.
            edge = (num / den) if den > 1e-9 else 0.0  # -1 ... +1
            edge *= fator * quality
            p_up = 0.5 + cfg.max_edge[i] * max(-1.0, min(1.0, edge))
            out.append(HorizonProb(h, p_up))
        return out

    # ------------------------------------------------------------ internals

    def _evaluate(self, r: ChartReading) -> Decision:
        cfg = self.cfg
        if not r.ok:
            return Decision(INCERTA, reasons=list(r.reasons) or ["leitura incerta"], timeframe_s=r.timeframe_s)
        ctx = MarketContext.from_reading(r)
        if ctx is None:
            return Decision(INCERTA, reasons=["velas insuficientes na tela"], timeframe_s=r.timeframe_s)

        mods = run_all(ctx)
        n_closed = len(ctx.closed)
        quality = min(1.0, n_closed / 30.0)
        hs = self.probabilities(mods, quality)
        best = max(hs, key=lambda h: (h.prob, -h.candles))
        phase = candle_phase(ctx.elapsed_frac)
        d = Decision(NAO_ENTRAR, horizons=hs, best=best, modules=mods, phase=phase,
                     seconds_to_enter=r.purchase_seconds, timeframe_s=r.timeframe_s, decimals=ctx.decimals)

        sign = 1 if best.p_up >= 0.5 else -1
        d.agree = sum(1 for m in mods if m.weighted * sign >= cfg.agree_threshold)
        against = sum(1 for m in mods if m.weighted * sign <= -2 * cfg.agree_threshold)

        d.target = self._target(ctx, mods, best, sign)
        d.exit_mode = f"segurar até o vencimento ({d.expiry_text(best)})"
        d.expiry_hint = self._expiry_hint(r, best)

        reasons = d.reasons
        if r.payout_pct is None:
            reasons.append("payout não lido: sem ele não dá para saber o acerto mínimo")
        else:
            d.breakeven = breakeven(r.payout_pct)
            d.threshold = d.breakeven + cfg.margin + cfg.phase_margin.get(phase or "fim", 0.02)
            if best.prob < d.threshold:
                reasons.append(f"probabilidade {pct(best.prob)} abaixo do mínimo {pct(d.threshold)}")
        if r.purchase_seconds is None:
            reasons.append("tempo de compra não lido")
        elif r.purchase_seconds < cfg.min_seconds_to_enter:
            reasons.append("tempo de compra acabando")
        if n_closed < cfg.min_closed_candles:
            reasons.append(f"poucas velas na tela ({n_closed})")
        if d.agree < cfg.min_agree:
            reasons.append(f"só {d.agree} módulo(s) concordam (mínimo {cfg.min_agree})")
        if against > cfg.max_strong_against:
            reasons.append(f"{against} módulos apontam forte para o outro lado")
        if not reasons:
            d.action = best.direction
        return d

    def _stabilize(self, d: Decision) -> Decision:
        """Só acende COMPRA/VENDA depois de N leituras seguidas iguais (evita piscar)."""
        key = d.action if d.signal else None
        if key is not None and key == self._last_action:
            self._streak += 1
        else:
            self._streak = 1 if key is not None else 0
        self._last_action = key
        if d.signal and self._streak < self.cfg.stable_readings:
            d.reasons.append("confirmando o sinal na próxima leitura")
            d.action = NAO_ENTRAR
        return d

    @staticmethod
    def _target(ctx: MarketContext, mods: list[ModuleResult], best: HorizonProb, sign: int) -> float:
        """Preço de referência: ATR proporcional ao prazo, limitado pelo nível mais próximo."""
        move = 0.5 * ctx.atr * math.sqrt(best.candles)
        tgt = ctx.price + sign * move
        for m in mods:
            if m.key == "sr":
                lvl = m.details.get("resistencia" if sign > 0 else "suporte")
                if lvl is not None and (lvl - ctx.price) * sign > 0 and abs(lvl - ctx.price) < move:
                    tgt = lvl
        return round(tgt, ctx.decimals)

    @staticmethod
    def _expiry_hint(r: ChartReading, best: HorizonProb) -> str:
        if r.expiry_seconds is None or not r.timeframe_s:
            return ""
        want = best.candles * r.timeframe_s
        label = r.expiry_label or ""
        if abs(r.expiry_seconds - want) <= 0.5 * r.timeframe_s:
            return f"a expiração escolhida na IQ ({label}) bate com a sugerida"
        return (f"na IQ a expiração está em {label} (faltam {fmt_duration(r.expiry_seconds)}); "
                f"a sugerida é {fmt_duration(want)}")
