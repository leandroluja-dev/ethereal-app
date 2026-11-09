"""Detecção das velas por cor, direto nos pixels da captura.

A ideia é simples e não depende de posição fixa na tela:

1. Separa os pixels verdes e vermelhos saturados (HSV).
2. Cada vela vira um componente conectado (corpo + pavio, mesma cor).
3. Velas de verdade têm largura parecida e estão espaçadas por um passo
   regular. Textos, botões e ícones coloridos não formam essa sequência,
   então ficam de fora.
4. Para cada vela, o corpo são as linhas largas e o pavio as linhas finas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..models import Box, PixelCandle


@dataclass
class ColorConfig:
    """Faixas HSV (OpenCV: H 0-180, S e V 0-255) das velas no tema escuro."""

    green_h: tuple[int, int] = (35, 95)
    # vermelho: H <= red_h_low ou H >= red_h_high. Vai até 22 porque a vela atual
    # da IQ Option tem um brilho verde por cima que deixa o vermelho alaranjado.
    red_h_low: int = 22
    red_h_high: int = 165
    min_s: int = 70
    min_v: int = 70


@dataclass
class CandleDetection:
    candles: list[PixelCandle] = field(default_factory=list)
    pitch: float = 0.0
    body_width: int = 0
    reasons: list[str] = field(default_factory=list)
    # Sequências de velas à direita da principal (indicam vela não detectada).
    trailing_fragments: int = 0
    green_hsv: tuple[int, int, int] | None = None
    red_hsv: tuple[int, int, int] | None = None

    @property
    def box(self) -> Box | None:
        if not self.candles:
            return None
        x1 = min(c.body_left for c in self.candles)
        x2 = max(c.body_right for c in self.candles) + 1
        y1 = min(c.high_y for c in self.candles)
        y2 = max(c.low_y for c in self.candles) + 1
        return Box(x1, y1, x2 - x1, y2 - y1)


@dataclass
class _Blob:
    color: str
    x: int
    y: int
    w: int
    h: int
    labels: list[int]

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0


def color_masks(bgr: np.ndarray, cfg: ColorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Máscaras (0/255) dos pixels verdes e vermelhos saturados, e a imagem em HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = cfg.min_s, cfg.min_v
    green = cv2.inRange(hsv, (cfg.green_h[0], s, v), (cfg.green_h[1], 255, 255))
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, s, v), (cfg.red_h_low, 255, 255)),
        cv2.inRange(hsv, (cfg.red_h_high, s, v), (180, 255, 255)),
    )
    return green, red, hsv


def _remove_long_lines(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Tira linhas horizontais longas e finas (linha de preço, botões largos)."""
    if min_len < 3:
        return mask
    # Só as linhas da imagem com pixels suficientes podem conter uma linha longa.
    rows = np.nonzero(cv2.reduce(mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).ravel() >= 255 * min_len)[0]
    if rows.size == 0:
        return mask
    out = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    groups = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    for g in groups:
        y1, y2 = max(0, int(g[0]) - 3), min(mask.shape[0], int(g[-1]) + 4)
        band = mask[y1:y2]
        lines = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
        if not lines.any():
            continue
        cut = band.copy()
        cut[lines > 0] = 0
        out[y1:y2] = cut
    return out  # cortes deixados aqui são fechados em _fill_occlusions


def _fill_occlusions(mask: np.ndarray) -> np.ndarray:
    """Fecha cortes finos (1-2 px) feitos por linhas tracejadas desenhadas por cima do pavio."""
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vk)


def _blobs(mask: np.ndarray, color: str, max_w: int) -> tuple[list[_Blob], np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs: list[_Blob] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > max_w or area < 2:
            continue
        blobs.append(_Blob(color, int(x), int(y), int(w), int(h), [i]))
    return _merge_vertical(blobs), labels


def _merge_vertical(blobs: list[_Blob]) -> list[_Blob]:
    """Junta pedaços da mesma vela (mesma coluna, separados por poucos pixels)."""
    blobs = sorted(blobs, key=lambda b: (b.x, b.y))
    merged: list[_Blob] = []
    for b in blobs:
        joined = False
        for m in merged:
            ox = min(m.x + m.w, b.x + b.w) - max(m.x, b.x)
            if ox <= 0 or ox < 0.5 * min(m.w, b.w):
                continue
            gap = max(b.y - (m.y + m.h), m.y - (b.y + b.h))
            if gap <= 4:
                x1, y1 = min(m.x, b.x), min(m.y, b.y)
                x2, y2 = max(m.x + m.w, b.x + b.w), max(m.y + m.h, b.y + b.h)
                m.x, m.y, m.w, m.h = x1, y1, x2 - x1, y2 - y1
                m.labels.extend(b.labels)
                joined = True
                break
        if not joined:
            merged.append(_Blob(b.color, b.x, b.y, b.w, b.h, list(b.labels)))
    return merged


def _profile(blob: _Blob, labels: np.ndarray) -> np.ndarray:
    sub = labels[blob.y : blob.y + blob.h, blob.x : blob.x + blob.w]
    m = np.isin(sub, blob.labels)
    return m


def _body_width(m: np.ndarray) -> int:
    widths = m.sum(axis=1)
    if widths.size == 0:
        return 0
    return int(widths.max())


def _measure(blob: _Blob, m: np.ndarray, expected_bw: int) -> PixelCandle:
    widths = m.sum(axis=1)
    thr = max(2, int(np.ceil(0.6 * expected_bw))) if expected_bw >= 3 else max(1, expected_bw)
    body_rows = np.nonzero(widths >= thr)[0]
    certain = body_rows.size > 0
    if not certain:
        # Corpo encoberto (ex.: linha do preço por cima de um doji): usa a linha mais larga.
        wmax = widths.max()
        body_rows = np.nonzero(widths == wmax)[0]
        if wmax <= 1:
            mid = int(np.nonzero(widths > 0)[0].mean())
            body_rows = np.array([mid])
    top, bottom = int(body_rows.min()), int(body_rows.max())
    cols = np.nonzero(m[body_rows].any(axis=0))[0] if certain else np.nonzero(m.any(axis=0))[0]
    rows_any = np.nonzero(widths > 0)[0]
    return PixelCandle(
        x_center=blob.x + (cols.min() + cols.max()) / 2.0,
        body_left=blob.x + int(cols.min()),
        body_right=blob.x + int(cols.max()),
        high_y=blob.y + int(rows_any.min()),
        low_y=blob.y + int(rows_any.max()),
        body_top_y=blob.y + top,
        body_bottom_y=blob.y + bottom,
        color=blob.color,
        body_certain=certain,
    )


def _mode(values: list[int]) -> int:
    if not values:
        return 0
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    return int(vals[np.argmax(counts)])


def _chains(items: list[tuple[float, int, int, int]], pitch: float, tol: float,
            max_vgap: float) -> list[list[int]]:
    """Agrupa velas em sequências com passo ~constante e continuidade vertical.

    items: (cx, idx, y_topo, y_base). Velas vizinhas ficam perto na vertical
    (a abertura de uma é quase o fechamento da anterior), então um texto
    colorido longe do gráfico não entra na sequência.
    """
    items = sorted(items)
    best_prev: dict[int, int] = {}
    length: dict[int, int] = {}
    xs = [it[0] for it in items]
    for j in range(len(items)):
        length[j] = 1
        best_prev[j] = -1
        for i in range(j - 1, -1, -1):
            d = xs[j] - xs[i]
            if d > pitch + tol:
                break
            if abs(d - pitch) > tol:
                continue
            vgap = max(items[j][2] - items[i][3], items[i][2] - items[j][3], 0)
            if vgap > max_vgap:
                continue
            if length[i] + 1 > length[j]:
                length[j] = length[i] + 1
                best_prev[j] = i
    chains: list[list[int]] = []
    used: set[int] = set()
    for j in sorted(range(len(items)), key=lambda k: (-length[k], -xs[k])):
        if j in used:
            continue
        chain = []
        k = j
        while k != -1 and k not in used:
            chain.append(k)
            used.add(k)
            k = best_prev[k]
        chains.append([items[k][1] for k in reversed(chain)])
    return chains


def _dashed_columns(blobs: list[_Blob], widths: list[int]) -> set[int]:
    """Blobs que formam linhas verticais tracejadas (vários traços finos na mesma coluna)."""
    cols: dict[int, list[int]] = {}
    for i, (b, w) in enumerate(zip(blobs, widths)):
        if w <= 2:
            cols.setdefault(int(round(b.cx)), []).append(i)
    out: set[int] = set()
    for x, idx in cols.items():
        near = idx + cols.get(x - 1, []) + cols.get(x + 1, [])
        if len(set(near)) >= 3:
            out.update(near)
    return out


def _extend(main: list[int], blobs: list[_Blob], widths: list[int], max_w: int,
            pitch: float, tol: float, max_vgap: float) -> list[int]:
    """Estende a sequência para os lados aceitando velas estreitas (doji, só pavio).

    A vela que acabou de abrir costuma ser um doji coberto pela linha do preço;
    sem isso a vela anterior seria tomada como a atual.
    """
    main = list(main)
    used = set(main)
    dashed = _dashed_columns(blobs, widths)
    # Linha vertical contínua (hora da compra/expiração) é bem mais alta que qualquer vela.
    max_h = 1.3 * max(blobs[i].h for i in main)

    def find_at(x: float, ref: _Blob) -> int | None:
        best: tuple[float, int] | None = None
        for i, b in enumerate(blobs):
            if i in used or i in dashed or widths[i] > max_w or b.h > max_h:
                continue
            if max(b.y - (ref.y + ref.h), ref.y - (b.y + b.h), 0) > max_vgap:
                continue
            d = abs(b.cx - x)
            if d <= tol and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    for direction in (1, -1):
        while True:
            ref = main[-1] if direction == 1 else main[0]
            i = find_at(blobs[ref].cx + direction * pitch, blobs[ref])
            if i is None:
                break
            used.add(i)
            if direction == 1:
                main.append(i)
            else:
                main.insert(0, i)
    return main


def detect_candles(
    bgr: np.ndarray,
    cfg: ColorConfig | None = None,
    roi: Box | None = None,
    min_candles: int = 10,
) -> CandleDetection:
    """Acha as velas na imagem BGR. ``roi`` limita a busca (coordenadas da imagem)."""
    cfg = cfg or ColorConfig()
    ox, oy = 0, 0
    img = bgr
    if roi is not None:
        img = bgr[roi.y : roi.y2, roi.x : roi.x2]
        ox, oy = roi.x, roi.y
    det = CandleDetection()
    if img.size == 0:
        det.reasons.append("região do gráfico vazia")
        return det

    green, red, hsv = color_masks(img, cfg)
    W = img.shape[1]
    max_w = max(8, int(0.045 * max(W, 1280)))
    line_len = max(max_w + 2, 40)
    green = _fill_occlusions(_remove_long_lines(green, line_len))
    red = _fill_occlusions(_remove_long_lines(red, line_len))

    gb, gl = _blobs(green, "verde", max_w)
    rb, rl = _blobs(red, "vermelha", max_w)
    blobs = gb + rb
    masks = [_profile(b, gl if b.color == "verde" else rl) for b in blobs]
    widths = [_body_width(m) for m in masks]

    tall = [w for w, b in zip(widths, blobs) if b.h >= 3 and w >= 2]
    if len(tall) < min_candles:
        det.reasons.append("poucas velas coloridas na tela")
        return det
    bw = _mode(tall)
    wtol = max(1, int(round(0.3 * bw)))
    cand = [i for i, w in enumerate(widths) if abs(w - bw) <= wtol]
    if len(cand) < min_candles:
        det.reasons.append("larguras de vela inconsistentes")
        return det

    xs = sorted(blobs[i].x + blobs[i].w / 2.0 for i in cand)
    diffs = np.diff(xs)
    diffs = diffs[(diffs >= bw) & (diffs <= 4 * bw + 4)]
    if diffs.size == 0:
        det.reasons.append("espaçamento entre velas não encontrado")
        return det
    pitch = float(np.median(diffs))
    tol = max(1.0, 0.2 * pitch)

    heights = sorted(blobs[i].h for i in cand)
    max_vgap = max(4.0 * heights[len(heights) // 2], 0.06 * img.shape[0])
    chains = _chains([(blobs[i].cx, i, blobs[i].y, blobs[i].y + blobs[i].h) for i in cand],
                     pitch, tol, max_vgap)
    chains = [c for c in chains if len(c) >= 3]
    if not chains:
        det.reasons.append("sequência de velas não encontrada")
        return det
    main = max(chains, key=lambda c: (len(c), blobs[c[-1]].cx))
    # Refina o passo com a sequência principal.
    mx = [blobs[i].cx for i in main]
    if len(mx) >= 2:
        pitch = float((mx[-1] - mx[0]) / (len(mx) - 1))

    main = _extend(main, blobs, widths, bw + wtol, pitch, tol, max_vgap)
    if len(main) < min_candles:
        det.reasons.append(f"só {len(main)} velas em sequência")

    end_x = blobs[main[-1]].cx
    main_top = min(blobs[i].y for i in main)
    main_bot = max(blobs[i].y + blobs[i].h for i in main)
    trailing = 0
    for c in chains:
        if c is main or len(c) < 3:
            continue
        sx = blobs[c[0]].cx
        cy_top = min(blobs[i].y for i in c)
        cy_bot = max(blobs[i].y + blobs[i].h for i in c)
        overlaps = cy_bot > main_top and cy_top < main_bot
        if overlaps and end_x < sx <= end_x + 6 * pitch:
            trailing += 1
    det.trailing_fragments = trailing
    if trailing:
        det.reasons.append("vela não detectada perto da vela atual")

    out: list[PixelCandle] = []
    for i in main:
        pc = _measure(blobs[i], masks[i], bw)
        out.append(
            PixelCandle(
                x_center=pc.x_center + ox,
                body_left=pc.body_left + ox,
                body_right=pc.body_right + ox,
                high_y=pc.high_y + oy,
                low_y=pc.low_y + oy,
                body_top_y=pc.body_top_y + oy,
                body_bottom_y=pc.body_bottom_y + oy,
                color=pc.color,
                body_certain=pc.body_certain,
            )
        )
    det.candles = out
    det.pitch = pitch
    det.body_width = bw
    if bw < 3:
        det.reasons.append("velas muito finas para separar corpo e pavio: aumente o zoom do gráfico")

    # Cor típica (para o modo debug e futuros ajustes automáticos).
    for color, mask in (("verde", green), ("vermelha", red)):
        sel = hsv[::4, ::4][mask[::4, ::4] > 0]
        if sel.size:
            med = tuple(int(v) for v in np.median(sel, axis=0))
            if color == "verde":
                det.green_hsv = med  # type: ignore[assignment]
            else:
                det.red_hsv = med  # type: ignore[assignment]
    return det


def measure_forming(bgr: np.ndarray, x_center: float, body_w: int, label_y: float,
                    y_lo: int, y_hi: int, max_h: float = 1e9) -> PixelCandle | None:
    """Mede a vela atual usando a etiqueta do preço como referência do fechamento.

    A IQ Option desenha um brilho verde (um ponto) em cima do preço atual, que
    muda a cor da vela atual (vermelho vira laranja). Aqui aceitamos qualquer
    cor quente ou verde saturada na coluna da vela, sem azul/roxo (médias e RSI)
    nem amarelo puro (média móvel). O fechamento é a etiqueta; a abertura é a
    ponta do corpo mais longe dela.
    """
    h = bgr.shape[0]
    half = max(1, body_w // 2)
    x1, x2 = int(round(x_center - half)), int(round(x_center + half)) + 1
    y_lo, y_hi = max(0, y_lo), min(h, y_hi)
    if x2 <= x1 or y_hi <= y_lo:
        return None
    hsv = cv2.cvtColor(bgr[y_lo:y_hi, x1:x2], cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Amarelo puro (H 26-31 e bem saturado) é a média móvel; o corpo vermelho sob o
    # brilho verde também fica amarelado, mas menos saturado (S ~140-170).
    yellow_ma = (hh >= 26) & (hh <= 31) & (ss >= 190)
    ok = (ss >= 90) & (vv >= 90) & ~yellow_ma & ~((hh >= 88) & (hh <= 160))
    # Colunas coloridas no topo e na base do painel ao mesmo tempo são a linha
    # vertical da hora da compra (tracejada) passando pela vela: saem da conta.
    band = max(3, int(0.15 * ok.shape[0]))
    line_cols = ok[:band].any(axis=0) & ok[-band:].any(axis=0) & (ok.sum(axis=0) > 0.25 * ok.shape[0])
    ok[:, line_cols] = False
    cover = ok.sum(axis=1)
    ly = int(round(label_y - 0.5)) - y_lo
    if not (0 <= ly < len(cover)):
        return None
    # corpo: linhas com cobertura de pelo menos metade da largura
    body = cover >= max(2, int(0.5 * (x2 - x1)))
    near = [r for r in range(max(0, ly - 3), min(len(cover), ly + 4)) if body[r]]
    if near:
        r0 = r1 = near[0]
        while r0 - 1 >= 0 and body[r0 - 1]:
            r0 -= 1
        while r1 + 1 < len(cover) and body[r1 + 1]:
            r1 += 1
    else:
        # sem corpo encostando no preço: doji, desde que haja pavio passando ali
        if not any(cover[max(0, ly - 3): ly + 4] > 0):
            return None
        r0 = r1 = ly
    # pavio: segue para cima e para baixo enquanto houver pixels (tolera falhas de 3 px)
    def walk(r: int, step: int) -> int:
        gap = 0
        last = r
        while 0 <= r + step < len(cover):
            r += step
            if cover[r] > 0:
                last, gap = r, 0
            else:
                gap += 1
                if gap > 3:
                    break
        return last
    hi_r, lo_r = walk(r0, -1), walk(r1, 1)
    if lo_r - hi_r + 1 > max_h:
        return None  # alto demais: é uma linha vertical (hora da compra), não vela
    # abertura: ponta do corpo mais longe da etiqueta
    if abs(r0 - ly) >= abs(r1 - ly):
        open_r, color = r0, "vermelha"  # abriu em cima, preço desceu
    else:
        open_r, color = r1, "verde"
    if r0 == r1:
        color = "verde"
    top, bot = (r0, r1)
    return PixelCandle(
        x_center=float(x_center), body_left=x1, body_right=x2 - 1,
        high_y=y_lo + hi_r, low_y=y_lo + lo_r,
        body_top_y=y_lo + top, body_bottom_y=y_lo + bot,
        color=color, body_certain=r0 != r1,
    )
