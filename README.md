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

Depois da limpeza: 26.195.700 avaliações de 555.414 usuários (99,83% da matriz usuário × jogo vazia).

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

Requisitos: [Python](https://www.python.org/downloads/) 3.11 ou mais novo, [Git](https://git-scm.com/downloads)
e cerca de 8 GB de RAM livre. Todos os comandos são rodados na pasta do projeto: no Windows, no
PowerShell; no Linux e no macOS, no Terminal.

### 1. Baixar o projeto

```bash
git clone https://github.com/LucaWBohnenberger/Trabalho-Final-de-Recomendacao.git
cd Trabalho-Final-de-Recomendacao
```

### 2. Criar o ambiente e instalar as dependências

**Linux e macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell recusar o `Activate.ps1` por política de execução, rode antes
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e confirme.

Com o ambiente ativo, o comando do Python é `python` nos três sistemas. Em um terminal novo,
ative o ambiente de novo (segunda linha do bloco acima) antes de rodar os passos seguintes.

### 3. Colocar os dados

Os dados não ficam no Git (são grandes demais). Entre na página
[jvanelteren/boardgamegeek-reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews)
do Kaggle (precisa de login), baixe só estes dois arquivos e descompacte se vierem em `.zip`:

- `bgg-26m-reviews.csv` (2,2 GB)
- `games_detailed_info2025.csv` (110 MB)

Crie a pasta `data/raw` dentro do projeto e coloque os dois arquivos nela:

```
Trabalho-Final-de-Recomendacao/
  data/
    raw/
      bgg-26m-reviews.csv
      games_detailed_info2025.csv
```

### 4. Limpar os dados (uma vez, cerca de 2 minutos)

```bash
python src/clean.py
```

Gera `data/clean/games.csv` e `data/clean/reviews.csv`.

### 5. Abrir a interface

```bash
python src/app.py
```

Quando aparecer `Pronto: http://127.0.0.1:8000`, abra esse endereço no navegador. As três abas
são os três cenários. Para parar o servidor, use Ctrl+C no terminal.

A primeira execução de `app.py`, `recommend.py`, `describe.py` ou `evaluate.py` monta um cache das
matrizes esparsas em `data/clean/ratings_cache.npz` (menos de 1 minuto); as seguintes carregam
em segundos.

### Outros comandos (opcionais)

Recomendar direto no terminal, sem interface:

```bash
python src/recommend.py Torsten --top 10
```

Refazer as estatísticas descritivas (`resultados/estatisticas.json`):

```bash
python src/describe.py
```

Refazer a avaliação (`resultados/avaliacao.json`), descrita na próxima seção:

```bash
python src/evaluate.py
```

A avaliação roda em paralelo com `fork`, que só existe no Linux e no macOS. No Windows, rode-a
pelo [WSL](https://learn.microsoft.com/pt-br/windows/wsl/install). A interface, a limpeza e as
estatísticas funcionam nos três sistemas.

## Avaliação

`python src/evaluate.py` esconde notas de usuários sorteados, refaz a matriz sem elas e mede
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
