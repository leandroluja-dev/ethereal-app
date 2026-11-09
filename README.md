# CandleSense

App para macOS que lê o gráfico da IQ Option na tela, em tempo real, e mostra
a direção mais provável do próximo movimento (COMPRA, VENDA ou NÃO ENTRAR),
com a probabilidade, a expiração, o tempo que falta para entrar e um alvo de
referência.

**O CandleSense só lê pixels da tela.** Ele nunca clica, nunca move o mouse,
nunca envia teclas e nunca abre operações. Não usa APIs não oficiais da IQ
Option, então funciona igual com ativos OTC.

## Estado atual: Fase 2 (sinais, motor por regras e painel)

| Parte | Onde |
|---|---|
| Indicadores (EMA, RSI, ATR, Bollinger, topos/fundos, níveis) | `candlesense/signals/indicators.py` |
| Módulos de sinal, cada um com nota de -1 a +1, confiança e explicação | `candlesense/signals/modules.py` |
| Motor por regras: probabilidade por expiração, equilíbrio do payout, janela de entrada | `candlesense/engine/rules.py` |
| Painel sempre por cima | `candlesense/ui/panel.py` |
| Leitura de "PURCHASE TIME", expiração escolhida, relógio da IQ e tempo da vela | `candlesense/vision/layout.py`, `reader.py` |

### Módulos

| Módulo | O que olha |
|---|---|
| Tendência | média rápida (9) x lenta (21), inclinação, topos e fundos subindo/descendo |
| Suporte e resistência | níveis com 2+ toques; preço perto de um nível; rompimentos com corpo |
| Price action | engolfo, martelo, estrela cadente, vela de força, três velas seguidas, doji |
| Momento | RSI 14 e variação das últimas 3 velas; RSI extremo vira risco de reversão |
| Volatilidade | tamanho das velas recentes x o normal; fora das bandas de Bollinger. Mercado parado ou agitado demais reduz a confiança de tudo |
| Vela atual | corpo, posição do preço na vela, pavios; confiança cresce com o tempo passado da vela |

Distâncias de preço são medidas em ATR, então valem igual em qualquer ativo.

### Como o motor decide

1. Leitura incerta: mostra `LEITURA INCERTA` e não sinaliza.
2. Para 1, 2, 3 e 5 velas, faz a média das notas (pesada pelo prazo e pela
   confiança). A vela atual pesa mais em 1 vela; a tendência, em 5.
3. Converte em probabilidade com teto: no máximo 60% enquanto as regras não
   forem medidas contra o resultado real (placar da Fase 3, calibração na Fase 4).
4. Só sinaliza se **tudo** isto vale:
   - probabilidade acima do equilíbrio `1 / (1 + payout)` (86% -> 53,8%) mais a
     margem (`--margem`, padrão 2 pontos) e mais 1 ponto no meio da vela ou 2 no fim
     (o começo da vela é o momento preferido);
   - "PURCHASE TIME" lido e com pelo menos 5 s;
   - pelo menos 3 módulos concordam e no máximo 1 aponta forte para o outro lado;
   - 20+ velas fechadas na tela;
   - a mesma decisão em 2 leituras seguidas (não pisca).
5. Senão, `NÃO ENTRAR` com os motivos.

A expiração sugerida é a de maior probabilidade. O painel compara com a
expiração escolhida na IQ (lida da tela) e avisa se não bater. Saída: segurar
até o vencimento. Alvo: meio ATR por raiz do prazo, limitado pelo nível de
suporte/resistência mais próximo.

## Fase 1 (captura e leitura do gráfico)

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
  signals/             indicadores e módulos de sinal
  engine/rules.py      motor por regras (probabilidade, payout, janela de entrada)
  ui/panel.py          painel sempre por cima
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

## Instalação (macOS, Intel ou Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Uso

No Mac, dê dois cliques em **Abrir CandleSense.command** na pasta do projeto: ele
atualiza o código e abre o painel. (O arquivo é uma cópia de `tools/abrir_candlesense.command`.)

```bash
candlesense permissoes          # guia de Gravação de Tela (e --acessibilidade, opcional)
candlesense janelas             # mostra se achou a janela da IQ Option
candlesense                     # leitura ao vivo + painel (q ou Esc no painel fecha)
candlesense --debug             # idem + janela de debug
candlesense --salvar-debug dbg  # salva captura + debug a cada 5 s em ./dbg
candlesense capturar -n 5       # salva 5 capturas em ./captures (para virar teste)
candlesense debug captura.png   # lê uma captura salva e grava debug + painel + json em debug_out/
candlesense --sem-painel        # só o terminal
candlesense --margem 3          # exige 3 pontos acima do equilíbrio do payout
candlesense --painel-escala 1.3 # painel maior
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
