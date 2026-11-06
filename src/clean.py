#!/usr/bin/env python3
"""Limpeza da coleta 2025 do BoardGameGeek.

Entrada:  data/raw/games_detailed_info2025.csv, data/raw/bgg-26m-reviews.csv
Saida:    data/clean/games.csv, data/clean/reviews.csv (os originais nao sao alterados)

Uso:      python3 src/clean.py

So as colunas que o recomendador usa sao lidas e tratadas. Categorias e
mecanicas de games.csv estao em JSON (use json.loads).
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
GAMES_IN = BASE / "data" / "raw" / "games_detailed_info2025.csv"
REVIEWS_IN = BASE / "data" / "raw" / "bgg-26m-reviews.csv"
OUT_DIR = BASE / "data" / "clean"

CHUNK_SIZE = 1_000_000

GAME_COLS = [
    "id", "name", "yearpublished", "minplayers", "maxplayers", "playingtime", "minplaytime",
    "minage", "averageweight", "average", "bayesaverage", "Board Game Rank",
    "boardgamecategory", "boardgamemechanic",
]
# O BGG usa 0 para "nao informado" nestas colunas. Anos negativos sao reais (a.C.).
ZERO_IS_MISSING = ["yearpublished", "minplayers", "maxplayers", "playingtime", "minage", "averageweight"]
LIST_COLS = ["boardgamecategory", "boardgamemechanic"]
RATING_MIN, RATING_MAX = 1.0, 10.0


def literal_to_json(value):
    """Converte o repr de lista do Python para JSON; nulo continua nulo."""
    if pd.isna(value):
        return value
    return json.dumps(ast.literal_eval(value), ensure_ascii=False)


def clean_games():
    df = pd.read_csv(GAMES_IN, usecols=GAME_COLS)
    df = df.rename(columns={"id": "game_id", "Board Game Rank": "board_game_rank"})
    if df["game_id"].duplicated().any():
        raise ValueError("IDs de jogo repetidos em games_detailed_info2025.csv")

    df["name"] = df["name"].str.strip()
    for col in ZERO_IS_MISSING:
        df[col] = df[col].mask(df[col] == 0)

    # playingtime e igual ao tempo maximo; se o minimo passa dele, a duracao nao e confiavel.
    inverted = df["minplaytime"] > df["playingtime"]
    df.loc[inverted, "playingtime"] = np.nan
    df = df.drop(columns="minplaytime")

    for col in LIST_COLS:
        df[col] = df[col].map(literal_to_json)
    int_cols = ["board_game_rank"] + [c for c in ZERO_IS_MISSING if c != "averageweight"]
    df[int_cols] = df[int_cols].astype("Int64")

    df.to_csv(OUT_DIR / "games.csv", index=False)
    print(f"games: {len(df):,} jogos, {df.shape[1]} colunas, {int(inverted.sum())} duracoes invertidas viraram ausentes")
    return set(df["game_id"])


def clean_reviews(game_ids):
    stats = dict.fromkeys(
        ["read", "no_user", "bad_rating", "no_metadata", "duplicate", "written"], 0)
    hashes = []
    out = OUT_DIR / "reviews.csv"

    reader = pd.read_csv(
        REVIEWS_IN,
        usecols=["user", "rating", "ID"],
        dtype={"user": str},
        chunksize=CHUNK_SIZE,
    )
    for i, chunk in enumerate(reader):
        stats["read"] += len(chunk)
        chunk = chunk.rename(columns={"ID": "game_id"})[["user", "game_id", "rating"]]

        # A leitura padrao ja trata nomes como "null" como vazios; esses usuarios saem aqui.
        chunk["user"] = chunk["user"].str.strip().replace("", np.nan)
        keep = chunk["user"].notna()
        stats["no_user"] += int((~keep).sum())
        chunk = chunk[keep]

        keep = chunk["rating"].between(RATING_MIN, RATING_MAX)
        stats["bad_rating"] += int((~keep).sum())
        chunk = chunk[keep]

        keep = chunk["game_id"].isin(game_ids)
        stats["no_metadata"] += int((~keep).sum())
        chunk = chunk[keep]

        # Remover espacos do usuario pode fazer duas linhas coincidirem.
        pair = chunk["user"] + "|" + chunk["game_id"].astype(str)
        hashes.append(pd.util.hash_pandas_object(pair, index=False).values)

        chunk.to_csv(out, index=False, mode="w" if i == 0 else "a", header=i == 0)
        stats["written"] += len(chunk)

    all_hashes = np.concatenate(hashes)
    stats["duplicate"] = len(all_hashes) - len(np.unique(all_hashes))
    print("reviews:", {k: f"{v:,}" for k, v in stats.items()})
    if stats["duplicate"]:
        print("AVISO: pares usuario/jogo repetidos em reviews.csv")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    game_ids = clean_games()
    clean_reviews(game_ids)


if __name__ == "__main__":
    main()
