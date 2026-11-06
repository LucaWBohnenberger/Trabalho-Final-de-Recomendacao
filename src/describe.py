#!/usr/bin/env python3
"""Estatisticas descritivas da base limpa, usadas nos graficos da apresentacao.

Uso:      python3 src/describe.py

Le os mesmos dados do recomendador (recommend.py) e salva tudo em
resultados/estatisticas.json; o terminal mostra um resumo.
"""
import collections
import json

import numpy as np
import pandas as pd

import recommend as rec

OUT = rec.BASE / "resultados" / "estatisticas.json"

# Faixas de contagem; None no fim significa "sem limite".
USER_BANDS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 499), (500, None)]
YEAR_EDGES = [-5000, 1979, 1989, 1999, 2004, 2009, 2014, 2019, 2100]
YEAR_LABELS = ["<1980", "80s", "90s", "00-04", "05-09", "10-14", "15-19", "20+"]
TOP = 10


def band_counts(counts, bands):
    out = {}
    for low, high in bands:
        label = str(low) if low == high else f"{low}-{high}" if high else f"{low}+"
        out[label] = int(((counts >= low) & (counts <= (high or counts.max()))).sum())
    return out


def summary(values):
    return {"mediana": float(np.median(values)), "min": int(values.min()), "max": int(values.max())}


def describe_ratings(data, games):
    n_users, n_games = data.by_user.shape
    ratings = data.by_user.data.astype(np.float64)
    per_user = np.diff(data.by_user.indptr)
    per_game = np.diff(data.by_game.indptr)
    # np.round leva o ,5 para o par (7,5 vira 8 e 6,5 vira 6); aqui o ,5 sempre sobe.
    rounded = np.clip(np.floor(ratings + 0.5), rec.RATING_MIN, rec.RATING_MAX).astype(int)
    by_reviews = np.sort(per_game)[::-1]
    most_rated = np.argsort(-per_game)[:TOP]
    return {
        "usuarios": n_users,
        "jogos": n_games,
        "avaliacoes": int(data.by_user.nnz),
        "esparsidade": 1 - data.by_user.nnz / (n_users * n_games),
        "nota": {"media": float(ratings.mean()), "mediana": float(np.median(ratings)), "desvio": float(ratings.std()),
                 "fracao_inteiras": float(np.mean(ratings == np.round(ratings)))},
        "histograma_notas": {int(k): int(v) for k, v in zip(*np.unique(rounded, return_counts=True))},
        "por_usuario": summary(per_user),
        "por_jogo": summary(per_game),
        "faixas_usuario": band_counts(per_user, USER_BANDS),
        "fracao_no_top1pct_jogos": float(by_reviews[: len(by_reviews) // 100].sum() / by_reviews.sum()),
        "fracao_no_top10pct_jogos": float(by_reviews[: len(by_reviews) // 10].sum() / by_reviews.sum()),
        "usuarios_notas_iguais": int((data.stds < rec.EPS).sum()),
        "mais_avaliados": [
            {"jogo": games.at[int(data.game_ids[i]), "name"], "avaliacoes": int(per_game[i])} for i in most_rated
        ],
    }


def describe_games(games):
    categories = collections.Counter(c for values in games["boardgamecategory"] for c in values)
    mechanics = {m for values in games["boardgamemechanic"] for m in values}
    years = pd.cut(games["yearpublished"].dropna(), YEAR_EDGES, labels=YEAR_LABELS).value_counts().sort_index()
    return {
        "por_ano": {str(k): int(v) for k, v in years.items()},
        "complexidade_media": float(games["averageweight"].mean()),
        "duracao_mediana": float(games["playingtime"].median()),
        "categorias": len(categories),
        "mecanicas": len(mechanics),
        "top_categorias": categories.most_common(TOP),
    }


def main():
    data = rec.load_ratings()
    games = rec.load_game_info()
    stats = {"avaliacoes": describe_ratings(data, games), "jogos": describe_games(games)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    r, g = stats["avaliacoes"], stats["jogos"]
    print(f"{r['avaliacoes']:,} avaliacoes de {r['usuarios']:,} usuarios em {r['jogos']:,} jogos "
          f"({r['esparsidade']:.2%} da matriz vazia)")
    print(f"nota media {r['nota']['media']:.2f}, mediana {r['nota']['mediana']:.0f}, desvio {r['nota']['desvio']:.2f}")
    print(f"avaliacoes por usuario: mediana {r['por_usuario']['mediana']:.0f}; "
          f"por jogo: mediana {r['por_jogo']['mediana']:.0f}")
    print(f"{g['categorias']} categorias, {g['mecanicas']} mecanicas, complexidade media {g['complexidade_media']:.2f}")
    print(f"Resultados em {OUT}")


if __name__ == "__main__":
    main()
