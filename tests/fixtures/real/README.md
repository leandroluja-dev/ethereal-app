# Capturas reais

Coloque aqui capturas reais da janela da IQ Option (geradas com
`candlesense capturar`) junto de um gabarito `.json` com o mesmo nome:

```json
{
  "expect_ok": true,
  "payout": 87,
  "timer_seconds": 23,
  "tolerancia": 0.00002,
  "ultimas_velas": [
    {"open": 1.08312, "high": 1.08320, "low": 1.08301, "close": 1.08305}
  ]
}
```

`ultimas_velas` são as últimas velas do gráfico, da mais antiga para a atual,
conferidas à mão (passe o mouse sobre a vela na IQ Option para ver o OHLC).
Os campos são opcionais: o teste só confere o que estiver no arquivo.
