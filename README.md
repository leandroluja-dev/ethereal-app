# CandleSense

Assistente de decisão que **lê o gráfico de velas da IQ Option direto da tela** e mostra, num painel ao lado, o que está acontecendo no gráfico.

> O CandleSense **não clica, não opera, não move o mouse e não envia teclas** para a IQ Option. Ele só lê pixels da tela. Não usa nenhuma API da IQ Option.

## Status: Fase 1 concluída

| Fase | O que entrega | Status |
|---|---|---|
| 1 | Captura automática da janela, detecção do gráfico, leitura das velas, OCR e modo debug | ✅ pronta para você testar |
| 2 | Indicadores, módulos de sinal, motor por regras e painel completo | próxima |
| 3 | Registro de sinais, leitura automática do resultado e placar | |
| 4 | Modelo treinado e calibrado, ranking de expirações, modo de saída e janela de entrada | |
| 5 | Ajustes por ativo e horário, alertas e .exe final | |

Na fase 1 o painel mostra **SEM SINAL** de propósito: nenhuma probabilidade aparece antes de existir o motor que a calcula.

### O que a fase 1 já faz sozinha (sem configurar nada)

- Acha a janela da IQ Option pelo título: **app desktop ou navegador** (Chrome, Edge, Firefox…), em qualquer monitor, com qualquer escala do Windows.
- Captura só essa janela, 8 vezes por segundo. O painel fica **invisível para a captura**, então pode ficar por cima do gráfico sem atrapalhar.
- Encontra dentro da janela: a área das velas, o eixo de preço, o eixo de tempo, o payout e o nome do ativo. Redetecta quando a janela muda de tamanho ou de monitor.
- Descobre as cores de alta e de baixa (tema claro ou escuro).
- Lê cada vela (corpo e pavio) e converte pixel em preço pelos rótulos do eixo (regressão que descarta leituras de OCR erradas).
- Lê a etiqueta do preço atual, o cronômetro da vela, o payout e o ativo. Descobre o timeframe (M1, M5…) pelo eixo de tempo.
- Separa a vela em formação das fechadas.
- Ignora linhas sobrepostas (linha de preço, linhas de operação, médias móveis) e marca as velas que ficaram cobertas por algum indicador.
- Detecta troca de ativo e de timeframe.
- Cada leitura tem uma nota de confiança de 0 a 100%. Se a leitura falhar, o painel mostra **"Leitura incerta"** e não inventa nada.
- Se as velas estiverem finas demais, mostra **"Aumente o zoom do gráfico"**.
- Modo **Debug**: desenha por cima da captura tudo o que foi lido, para você conferir a olho.
- **Gravar sessão**: grava a tela para reprocessar depois, sem mercado aberto.

Tempo de cada leitura nos testes: ~80 ms (meta: abaixo de 300 ms). A primeira leitura, ou depois de mudar a janela, leva ~1 s porque procura o layout inteiro.

## Instalação (Windows)

1. Instale o **Python 3.11** (ou 3.12) em <https://www.python.org/downloads/>. Na primeira tela do instalador, marque **"Add python.exe to PATH"**. (O Python 3.13 ainda não tem a biblioteca de OCR.)
2. Baixe este projeto: no GitHub, abra o branch `claude/magical-babbage-zmewmq`, clique em **Code → Download ZIP** e extraia numa pasta (ex.: `C:\CandleSense`). Ou, com git:

   ```bash
   git clone -b claude/magical-babbage-zmewmq https://github.com/leandroluja-dev/ethereal-app.git C:\CandleSense
   ```

3. Dê dois cliques em **`instalar.bat`**. Ele cria um ambiente `.venv` e instala tudo (leva alguns minutos na primeira vez).

Instalação manual, se preferir:

```bash
py -3.11 -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

## Como usar

1. Abra a IQ Option (app ou navegador) num **gráfico de velas**.
2. Dê dois cliques em **`iniciar.bat`**.
3. O painel aparece. Clique em **Ajustar janela** para colocar a IQ Option e o painel lado a lado (opcional).

Botões do painel:

| Botão | O que faz |
|---|---|
| Ajustar janela | Reposiciona/redimensiona a IQ Option pela API do Windows (`SetWindowPos`) e coloca o painel ao lado. Nunca usa mouse ou teclado. |
| Debug | Abre a janela que mostra a captura com as detecções desenhadas por cima. |
| Placar | Chega na fase 3. |
| Pausar | Para a leitura. |
| Gravar sessão | Grava a tela em `recordings/` (2 quadros por segundo) para reprocessar depois. |
| Salvar captura | Salva em `captures/` o print bruto, o print com as detecções e a leitura em JSON. |
| Redetectar | Procura de novo o gráfico, o eixo e o payout. |

### Legenda da janela de Debug

- **Ciano**: área das velas · **Amarelo**: eixo de preço · **Magenta**: eixo de tempo
- **Contorno verde/vermelho** em cada vela: corpo detectado; a linha é o pavio
- **Contorno branco grosso**: vela em formação
- **X magenta**: vela parcialmente coberta por indicador ou marcador (valores menos confiáveis)
- **Rótulos do eixo**: verde = usado no cálculo do preço; vermelho = descartado
- **Linhas ciano finas**: onde o app acha que está cada preço lido. Elas devem passar no meio dos rótulos do eixo. Se não passarem, a conversão pixel→preço está errada.
- Canto inferior esquerdo: resumo da leitura, confiança por parte e tempo gasto.

### Plano B: área manual

Se a detecção automática falhar, abra o **Debug**, clique em **Selecionar área das velas (plano B)** e arraste sobre a imagem marcando só a região das velas (sem o eixo de preço). A área fica salva em `candlesense_user.yaml`. **Voltar à detecção automática** desfaz.

## Como testar a fase 1 e o que me mandar

1. Abra a IQ Option no M1 de um ativo OTC e rode `iniciar.bat`.
2. Confira no painel: ativo, payout, timeframe, cronômetro e preço atual batem com a tela?
3. Abra o **Debug** e confira se os contornos caem em cima das velas e se as linhas ciano passam no meio dos rótulos do eixo.
4. Troque de ativo, de M1 para M5, mude o zoom e redimensione a janela. O painel deve acompanhar.
5. Teste também no navegador.
6. Clique em **Salvar captura** em 2 ou 3 situações (de preferência uma em que algo saiu errado) e me mande os arquivos de `captures/` e o log `logs/candlesense.log`.

Com esses prints eu calibro a visão para a tela real e transformo cada um em teste automático: basta colocar o print em `tests/fixtures/real/` com um `.json` ao lado com os valores corretos, por exemplo:

```json
{"status": "OK", "asset": "EUR/USD (OTC)", "payout": 87, "timeframe_min": 1, "current_price": 1.08452}
```

## Outros comandos

Demonstração com mercado sintético (não precisa da IQ Option; dois cliques em `demo.bat`):

```bash
.venv\Scripts\python -m candlesense demo
```

Analisar um print salvo (gera `captures/<nome>_debug.png` e `<nome>_leitura.json`):

```bash
.venv\Scripts\python -m candlesense analyze meu_print.png
```

Abrir o painel lendo um print fixo:

```bash
.venv\Scripts\python -m candlesense image meu_print.png
```

Reprocessar uma sessão gravada, sem tela (mostra quantos quadros deram OK e a latência):

```bash
.venv\Scripts\python -m candlesense replay recordings\20261109_153000
```

O mesmo, mostrando no painel:

```bash
.venv\Scripts\python -m candlesense replay recordings\20261109_153000 --gui
```

Rodar os testes automáticos (ou dois cliques em `testar.bat`):

```bash
.venv\Scripts\python -m pytest -q
```

## Configuração

Tudo fica em [`config.yaml`](config.yaml), com comentários. Os valores que estão lá já são os padrões: você não precisa mexer em nada. Os mais úteis:

- `vision.min_confidence`: abaixo disso a leitura vira "incerta" (padrão 0,6).
- `vision.assume_timeframe_min`: fixe `1` ou `5` se o eixo de tempo não puder ser lido.
- `vision.clock_offset_s`: ajuste se o relógio do PC estiver adiantado/atrasado em relação à corretora.
- `capture.fps`: quadros analisados por segundo.

## Estrutura

```
candlesense/
  __main__.py          linha de comando (run, demo, analyze, image, replay)
  app.py               monta o app: captura + análise + painel + debug
  config.py            padrões + config.yaml
  capture/
    window.py          acha a janela (pywin32), DPI, SetWindowPos, painel invisível à captura
    grabber.py         captura com mss; fontes de imagem fixa e demonstração
    layout.py          acha área das velas, eixos, payout e ativo dentro da janela
    frame.py           quadro capturado
  vision/
    colors.py          descobre as cores de alta/baixa e o tema
    candles.py         corpo e pavio de cada vela, vela em formação, oclusões
    price_axis.py      OCR do eixo de preço (pixel→preço), etiqueta do preço atual, cronômetro, eixo de tempo
    ocr.py             RapidOCR (padrão) ou Tesseract
    text.py            interpretação de preços, %, horários e nomes de ativos
    hud.py             releitura periódica de payout e ativo
    state.py           troca de ativo/timeframe
    analyzer.py        orquestra tudo e calcula a confiança
    debug_draw.py      desenho de debug
  replay/              gravação, reprodução e gerador de gráficos sintéticos
  ui/                  painel PyQt6, janela de debug e thread de análise
  series/ signals/ engine/ decision/ journal/   fases 2 a 4
tests/                 testes com capturas salvas (tests/fixtures) e gabarito
```

## Limitações conhecidas (a calibrar com seus prints)

- A visão foi desenvolvida e testada com gráficos sintéticos que imitam a IQ Option, porque eu não tenho acesso à sua tela. Os prints reais são o próximo passo para calibrar.
- O nome do ativo é lido da aba ativa no topo. Se houver muitas abas abertas, a escolha pode falhar; o painel mostra "—" em vez de chutar.
- Se nenhum cronômetro for lido na tela, o app calcula o tempo restante pelo relógio do PC (as velas fecham na virada do minuto). Mantenha o relógio do Windows sincronizado.
- Velas cobertas por indicadores são marcadas como "cobertas". Para a melhor leitura, deixe o gráfico sem indicadores sobrepostos às velas.
