# Recomendador de jogos de tabuleiro (BoardGameGeek)

Trabalho T1 da disciplina de Sistemas de Recomendação (Prof. Rafael Scopel, PUCRS).

**Integrantes:** Luca Wolffenbuttel Bohnenberger e Louise Zanol Northfleet

Sistema de **recomendação personalizada por vizinhos próximos** (kNN usuário-usuário) sobre
26 milhões de avaliações do BoardGameGeek, com interface web para os três cenários pedidos:

| Cenário | Técnica |
|---|---|
| Lista para um usuário antigo | kNN com Pearson ponderado por n/(n+25) e previsão por z-score; a ordem desconta o escore de produto (média simples acima da bayesiana) |
| Lista para um usuário novo | O mesmo kNN a partir das notas dadas na tela, com filtros de conhecimento (jogadores, duração, idade, complexidade); com menos de 5 notas, popularidade |
| Item novo para um usuário antigo | Conteúdo: Jaccard de categorias e mecânicas entre o jogo novo e os jogos que o usuário já avaliou |

A apresentação está em [`apresentacao/recomendador-bgg.pptx`](apresentacao/recomendador-bgg.pptx).

## Base de dados

[BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews), coleta de 2025:

- `bgg-26m-reviews.csv`: uma linha por avaliação (usuário, jogo, nota de 1 a 10, comentário);
- `games_detailed_info2025.csv`: metadados dos 27.780 jogos.

Depois da limpeza: 26.195.766 avaliações de 555.415 usuários (99,83% da matriz usuário × jogo vazia).

## Estrutura

```
src/
  clean.py       limpeza dos dois arquivos do Kaggle
  describe.py    estatísticas descritivas da base
  recommend.py   o recomendador (também roda no terminal)
  app.py         interface web dos três cenários
  evaluate.py    avaliação offline com hold-out
resultados/      estatísticas e métricas usadas na apresentação
apresentacao/    slides
data/            dados brutos e limpos (fora do Git)
```

## Como rodar

Requer Python 3.11+ e cerca de 8 GB de RAM.

```bash
pip install -r requirements.txt

mkdir -p data/raw
for f in bgg-26m-reviews.csv games_detailed_info2025.csv; do
  curl -L -o data/raw/$f "https://www.kaggle.com/api/v1/datasets/download/jvanelteren/boardgamegeek-reviews/$f"
done

python3 src/clean.py       # ~7 min; grava data/clean/
python3 src/describe.py    # grava resultados/estatisticas.json
python3 src/app.py         # http://127.0.0.1:8000
```

A primeira execução de `recommend.py`, `app.py` ou `evaluate.py` monta um cache das matrizes
esparsas em `data/clean/ratings_cache.npz` (cerca de 1 minuto); as seguintes carregam em segundos.

Também dá para recomendar direto no terminal:

```bash
python3 src/recommend.py Torsten --top 10
```

## Avaliação

`python3 src/evaluate.py` esconde notas de usuários sorteados, refaz a matriz sem elas e mede
o erro só no que foi escondido. Leva cerca de 30 minutos com 4 processos; `--secoes` roda
só algumas (`--secoes A,E`).

| Seção | Pergunta | Métricas |
|---|---|---|
| A | Quais hiperparâmetros (mínimo de jogos em comum, peso de significância)? | RMSE, MAE, cobertura |
| B | Quantas notas um usuário novo precisa dar? | RMSE, MAE |
| C | O conteúdo prevê bem um jogo sem avaliações? | RMSE, MAE |
| D | Quem recebe lista e quais jogos se repetem? | Cobertura de usuários e de catálogo |
| E | Modelo final | MAE, MSE e RMSE com IC 95%, Spearman, AUC da curva ROC, cobertura, diversidade |

Resultados do modelo final (seção E, 3.995 notas escondidas de 400 usuários):

| Método | MAE | RMSE (IC 95%) |
|---|---|---|
| Média do usuário | 1,064 | 1,406 (1,36–1,45) |
| Média do jogo | 1,024 | 1,355 (1,31–1,40) |
| **kNN** | **0,955** | **1,284 (1,24–1,32)** |

Para separar as notas 8 ou mais das demais, o kNN tem AUC de 0,79, contra 0,74 da média do jogo.
A correlação de Spearman média entre a ordem prevista e a real é de 0,48.
Os números completos estão em [`resultados/avaliacao.json`](resultados/avaliacao.json).
