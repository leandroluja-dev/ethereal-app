# CandleSense

App para macOS que lê o gráfico da IQ Option na tela, em tempo real, e (a partir
da Fase 2) mostra a direção mais provável do próximo movimento com probabilidade
calibrada.

**O CandleSense só lê pixels da tela.** Ele nunca clica, nunca move o mouse,
nunca envia teclas e nunca abre operações. Não usa APIs não oficiais da IQ
Option, então funciona igual com ativos OTC.

## Estado atual: Fase 1 (captura e leitura do gráfico)

| Parte | Onde | Situação |
|---|---|---|
| Permissões (Gravação de Tela, Acessibilidade) com guia na 1ª execução | `candlesense/platform/permissions.py` | pronto, testar no Mac |
| Achar a janela da IQ Option (app ou aba do navegador) | `candlesense/platform/windows.py` | pronto + testes |
| Capturar só essa janela (ScreenCaptureKit, Retina) | `candlesense/platform/capture.py` | pronto, testar no Mac |
| Ajuste da janela só por Acessibilidade (opcional) | `candlesense/platform/accessibility.py` | pronto, testar no Mac |
| Velas por cor (corpo, pavio, doji, vela atual) | `candlesense/vision/candles.py` | pronto + testes |
| Eixo de preço (OCR + ajuste robusto + grade) | `candlesense/vision/axis.py` | pronto + testes |
| Cronômetro, payout, etiqueta do preço atual | `candlesense/vision/layout.py`, `reader.py` | pronto + testes |
| OCR: Vision (Apple) com fallback rapidocr | `candlesense/vision/ocr.py` | rapidocr testado; Vision testar no Mac |
| Modo debug (desenha a detecção sobre a captura) | `candlesense/debug/overlay.py` | pronto |

## Estrutura

```
candlesense/
  __main__.py          linha de comando (ao vivo, debug, capturar, permissoes, janelas)
  models.py            Box, PixelCandle, Candle, PriceScale, Layout, ChartReading
  platform/            tudo que é específico do macOS (pyobjc)
    permissions.py     Gravação de Tela e Acessibilidade
    windows.py         acha a janela (CGWindowList)
    capture.py         ScreenCaptureKit + fallback CGWindowList + leitor de arquivos
    accessibility.py   ajuste da janela via AXUIElement
  vision/              leitura do gráfico (só numpy/OpenCV, roda em qualquer sistema)
    candles.py         velas por cor, sequência com passo regular
    axis.py            rótulos do eixo -> escala y -> preço
    layout.py          acha eixo, cronômetro e payout com um OCR da tela inteira
    ocr.py             Vision / rapidocr e leitura de números, mm:ss e %
    reader.py          junta tudo; diz "leitura incerta" quando algo não fecha
  debug/overlay.py     imagem de debug
tests/                 testes com capturas salvas e gabarito
tools/make_fixtures.py gera as capturas sintéticas de teste
```

## Como funciona a leitura

1. **Velas**: separa os pixels verdes e vermelhos saturados. Cada vela é um
   bloco de uma cor só (corpo + pavio). Só entram blocos com a mesma largura,
   espaçados por um passo regular e próximos na vertical; botões, textos
   coloridos e linhas tracejadas ficam de fora. Corpo = linhas largas, pavio =
   linhas finas.
2. **Layout** (uma vez, ou quando a janela/zoom muda): um OCR da tela inteira
   acha a coluna de preços à direita das velas, o cronômetro `mm:ss` perto da
   vela atual (rótulos de hora do eixo X são ignorados) e o payout `+87%`.
3. **Escala**: os rótulos do eixo são alinhados com as linhas da grade e
   ajustados por uma reta robusta (rótulo lido errado é descartado).
4. **Preço atual**: a etiqueta colorida no eixo é lida a cada quadro e precisa
   bater com o fechamento da vela atual. Se não bater, a leitura é incerta.
5. **Nunca inventa dados**: poucas velas, eixo ilegível, vela recente
   encoberta, etiqueta divergente ou erro interno viram `leitura incerta`, com
   o motivo. O leitor nunca derruba o app.

## Instalação (Mac com Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Uso

```bash
candlesense permissoes          # guia de Gravação de Tela (e --acessibilidade, opcional)
candlesense janelas             # mostra se achou a janela da IQ Option
candlesense --debug             # leitura ao vivo + janela de debug (q ou Esc fecha)
candlesense --salvar-debug dbg  # salva captura + debug a cada 5 s em ./dbg
candlesense capturar -n 5       # salva 5 capturas em ./captures (para virar teste)
candlesense debug captura.png   # lê uma captura salva e grava debug_out/captura_debug.png
```

Para testar a Fase 1: abra a IQ Option (app ou navegador) em 1920x1080 a 100%,
tema escuro, gráfico de velas, e rode `candlesense --debug`. A janela de debug
desenha cada vela, o eixo, o cronômetro e o payout; o topo mostra `OK` ou
`LEITURA INCERTA` com o motivo.

## Testes

```bash
pytest -q
```

As capturas em `tests/fixtures/synthetic` imitam o tema escuro (velas, eixo,
cronômetro, payout, botões coloridos, barra de navegador, Retina 2x, cripto
com 2 casas, doji coberto pela linha do preço, velas finas demais e tela sem
gráfico) e têm o OHLC exato como gabarito. Capturas reais entram em
`tests/fixtures/real` (veja o README de lá).
