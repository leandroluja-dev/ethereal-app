#!/bin/bash
# Dois cliques abrem o CandleSense no Terminal.
# Copie para a pasta do projeto como "Abrir CandleSense.command" (já vem assim no Mac).
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$DIR/pyproject.toml" ] || DIR="$(cd "$DIR/.." && pwd)"
cd "$DIR" || exit 1
clear
echo "== CandleSense =="
echo "Atualizando o código..."
git pull --ff-only -q 2>/dev/null || echo "(sem atualização agora, seguindo com a versão atual)"
source .venv/bin/activate
pip install -q -e . >/dev/null 2>&1
echo "Abrindo o painel. Para fechar: clique no painel e aperte q."
echo
candlesense "$@"
echo
read -n 1 -s -r -p "CandleSense fechado. Aperte qualquer tecla para fechar esta janela."
