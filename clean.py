#!/usr/bin/env python3
"""Limpeza da coleta 2025 do BoardGameGeek.

Entrada:  games_detailed_info2025.csv, bgg-26m-reviews.csv
Saida:    clean/games.csv, clean/reviews.csv (os originais nao sao alterados)

Uso:      python3 clean.py

Para ler as saidas sem transformar usuarios como "NA" ou "null" em valor nulo:

    pd.read_csv(path, keep_default_na=False, na_values=[""])

As colunas de lista e de enquete de games.csv estao em JSON (use json.loads).
"""
import ast
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
GAMES_IN = BASE / "games_detailed_info2025.csv"
REVIEWS_IN = BASE / "bgg-26m-reviews.csv"
OUT_DIR = BASE / "clean"

CHUNK_SIZE = 1_000_000
# So o campo vazio e nulo: "NA", "null" e "None" sao nomes de usuario e comentarios validos.
READ_OPTS = {"keep_default_na": False, "na_values": [""]}

# Sem informacao: Unnamed: 0 e um indice salvo, type e constante, median e sempre 0
# e os dois rankings tem um unico jogo preenchido.
GAME_DROP = ["Unnamed: 0", "type", "median", "RPG Item Rank", "Accessory Rank"]
# O BGG usa 0 para "nao informado" nestas colunas. Anos negativos sao reais (a.C.).
ZERO_IS_MISSING = [
    "yearpublished", "minplayers", "maxplayers", "playingtime",
    "minplaytime", "maxplaytime", "minage", "averageweight",
]
LITERAL_COLS = [
    "alternate", "suggested_num_players", "suggested_playerage",
    "suggested_language_dependence", "boardgamecategory", "boardgamemechanic",
    "boardgamefamily", "boardgameexpansion", "boardgameaccessory",
    "boardgamecompilation", "boardgameimplementation", "boardgamedesigner",
    "boardgameartist", "boardgamepublisher", "boardgameintegration",
]
RATING_MIN, RATING_MAX = 1.0, 10.0


def clean_text(s):
    """Decodifica entidades HTML e reduz qualquer espaco em branco a um espaco."""
    mask = s.notna()
    out = s[mask].map(html.unescape).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.copy()
    s[mask] = out
    return s.mask(s == "")


def literal_to_json(value):
    """Converte o repr de lista/dict do Python para JSON; nulo continua nulo."""
    if pd.isna(value):
        return value
    return json.dumps(ast.literal_eval(value), ensure_ascii=False)


def snake_rank(col):
    return col.lower().replace("'", "").replace(" ", "_")


def clean_games():
    df = pd.read_csv(GAMES_IN, low_memory=False, **READ_OPTS)
    df = df.drop(columns=GAME_DROP).rename(columns={"id": "game_id"})

    df["name"] = df["name"].str.strip()
    df["description"] = clean_text(df["description"])

    for col in ZERO_IS_MISSING:
        df[col] = df[col].mask(df[col] == 0)

    swapped = df["minplaytime"] > df["maxplaytime"]
    df.loc[swapped, ["minplaytime", "maxplaytime"]] = df.loc[
        swapped, ["maxplaytime", "minplaytime"]
    ].values
    # playingtime e igual a maxplaytime em todo o arquivo original.
    df.loc[swapped, "playingtime"] = df.loc[swapped, "maxplaytime"]

    for col in LITERAL_COLS:
        df[col] = df[col].map(literal_to_json)

    rank_cols = [c for c in df.columns if c.endswith("Rank")]
    df = df.rename(columns={c: snake_rank(c) for c in rank_cols})
    int_cols = [snake_rank(c) for c in rank_cols] + [c for c in ZERO_IS_MISSING if c != "averageweight"]
    df[int_cols] = df[int_cols].astype("Int64")

    # Primeiro as colunas de identificacao, na ordem em que costumam ser consultadas.
    first = ["game_id", "name", "yearpublished"]
    df = df[first + [c for c in df.columns if c not in first]]

    df.to_csv(OUT_DIR / "games.csv", index=False)
    print(f"games: {len(df):,} jogos, {df.shape[1]} colunas, {int(swapped.sum())} tempos de jogo invertidos")
    return set(df["game_id"])


def clean_reviews(game_ids):
    stats = dict.fromkeys(["read", "no_user", "bad_rating", "no_metadata", "duplicate", "written"], 0)
    hashes = []
    out = OUT_DIR / "reviews.csv"

    reader = pd.read_csv(
        REVIEWS_IN,
        usecols=["user", "rating", "comment", "ID"],
        dtype={"user": str, "comment": str},
        chunksize=CHUNK_SIZE,
        **READ_OPTS,
    )
    for i, chunk in enumerate(reader):
        stats["read"] += len(chunk)
        chunk = chunk.rename(columns={"ID": "game_id"})[["user", "game_id", "rating", "comment"]]

        chunk["user"] = chunk["user"].str.strip().replace("", np.nan)
        keep = chunk["user"].notna()
        stats["no_user"] += int((~keep).sum())
        chunk = chunk[keep]

        keep = chunk["rating"].between(RATING_MIN, RATING_MAX)
        stats["bad_rating"] += int((~keep).sum())
        chunk = chunk[keep]

        keep = chunk["game_id"].isin(game_ids)
        stats["no_metadata"] += int((~keep).sum())
        chunk = chunk[keep].copy()

        chunk["comment"] = clean_text(chunk["comment"])

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
    OUT_DIR.mkdir(exist_ok=True)
    game_ids = clean_games()
    clean_reviews(game_ids)


if __name__ == "__main__":
    main()
