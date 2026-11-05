#!/usr/bin/env python3
"""Recomendacao de jogos por filtragem colaborativa baseada em usuario.

Para um usuario de interesse:
  1. Calcula a correlacao de Pearson com quem tem ao menos 5 jogos em comum e
     multiplica por n / (n + 25), sendo n o numero de jogos em comum.
  2. Para cada jogo que ele ainda nao avaliou, escolhe os 10 vizinhos mais
     similares que avaliaram esse jogo.
  3. Faz a media ponderada das notas normalizadas (z-score) dos vizinhos e volta
     para a escala do usuario:

         nota = media_u + desvio_u * soma(sim_v * (r_v - media_v) / desvio_v) / soma(sim_v)

  4. Ordena por uma pontuacao que desconta da nota o escore de produto: quanto a
     media do jogo no BGG fica acima da media bayesiana, sinal de poucos
     avaliadores muito entusiasmados. A nota so e limitada a escala de 1 a 10
     depois de ordenar.

Usuario novo pode restringir a lista com filtros de conhecimento (jogadores,
duracao, idade e complexidade). Item novo usa conteudo: categorias e mecanicas.

Uso:
    python3 src/recommend.py                  # escolhe o usuario no terminal
    python3 src/recommend.py NOME --top 20    # recomenda direto para NOME

Requer data/clean/reviews.csv e data/clean/games.csv, gerados por clean.py. A
primeira execucao monta data/clean/ratings_cache.npz, que deixa as seguintes bem mais rapidas.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BASE = Path(__file__).resolve().parent.parent
CLEAN = BASE / "data" / "clean"
REVIEWS = CLEAN / "reviews.csv"
GAMES = CLEAN / "games.csv"
CACHE = CLEAN / "ratings_cache.npz"

# So o campo vazio e nulo: "null" e um nome de usuario valido.
READ_OPTS = {"keep_default_na": False, "na_values": [""]}

N_NEIGHBORS = 10
MIN_COMMON = 5
SIGNIFICANCE = 25  # quanto maior, mais o peso favorece quem tem muitos jogos em comum
PRODUCT_WEIGHT = 1.0  # pontos descontados por ponto de media acima da media bayesiana
RATING_MIN, RATING_MAX = 1.0, 10.0
EPS = 1e-9


@dataclass
class Ratings:
    by_user: sparse.csr_matrix
    by_game: sparse.csc_matrix
    users: np.ndarray  # nomes em ordem alfabetica, alinhados com as linhas
    game_ids: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def build_cache():
    """Le as avaliacoes do CSV e salva as matrizes esparsas e as estatisticas de usuario."""
    print("Montando o cache (so na primeira execucao)...")
    df = pd.read_csv(
        REVIEWS,
        usecols=["user", "game_id", "rating"],
        dtype={"user": str, "game_id": np.int32, "rating": np.float32},
        **READ_OPTS,
    )
    user_idx, users = pd.factorize(df["user"], sort=True)
    game_idx, game_ids = pd.factorize(df["game_id"], sort=True)
    by_user = sparse.csr_matrix(
        (df["rating"].to_numpy(), (user_idx, game_idx)),
        shape=(len(users), len(game_ids)),
    )
    by_game = by_user.tocsc()

    values = by_user.data.astype(np.float64)
    rows = np.repeat(np.arange(by_user.shape[0]), np.diff(by_user.indptr))
    counts = np.bincount(rows, minlength=by_user.shape[0])
    means = np.bincount(rows, weights=values, minlength=by_user.shape[0]) / counts
    stds = np.sqrt(np.bincount(rows, weights=(values - means[rows]) ** 2, minlength=by_user.shape[0]) / counts)

    np.savez(
        CACHE,
        user_indptr=by_user.indptr, user_indices=by_user.indices, user_data=by_user.data,
        game_indptr=by_game.indptr, game_indices=by_game.indices, game_data=by_game.data,
        users=np.array(users, dtype=str), game_ids=np.asarray(game_ids),
        means=means, stds=stds,
    )


def load_ratings():
    if not CACHE.exists() or CACHE.stat().st_mtime < REVIEWS.stat().st_mtime:
        build_cache()
    with np.load(CACHE) as z:
        shape = (len(z["users"]), len(z["game_ids"]))
        return Ratings(
            by_user=sparse.csr_matrix((z["user_data"], z["user_indices"], z["user_indptr"]), shape=shape),
            by_game=sparse.csc_matrix((z["game_data"], z["game_indices"], z["game_indptr"]), shape=shape),
            users=z["users"],
            game_ids=z["game_ids"],
            means=z["means"],
            stds=z["stds"],
        )


def load_game_info():
    """Metadados dos jogos, com "features" reunindo categorias e mecanicas."""
    games = pd.read_csv(
        GAMES,
        usecols=[
            "game_id", "name", "yearpublished", "board_game_rank", "boardgamecategory", "boardgamemechanic",
            "average", "bayesaverage", "minplayers", "maxplayers", "playingtime", "minage", "averageweight",
        ],
        **READ_OPTS,
    ).set_index("game_id")
    for col in ["boardgamecategory", "boardgamemechanic"]:
        games[col] = [json.loads(v) if isinstance(v, str) else [] for v in games[col]]
    games["features"] = [
        item_features(c, m) for c, m in zip(games["boardgamecategory"], games["boardgamemechanic"])
    ]
    return games


def item_features(categories, mechanics):
    return frozenset(categories) | frozenset(mechanics)


def product_penalty(data, games, weight=PRODUCT_WEIGHT):
    """Desconto de cada jogo (na ordem de data.game_ids) por ter media inflada.

    A media bayesiana do BGG puxa jogos com poucos votos para baixo; a diferenca
    para a media simples mede essa falta de confianca. Jogos com media abaixo
    da bayesiana nao ganham bonus.
    """
    gap = (games["average"] - games["bayesaverage"]).clip(lower=0)
    return weight * gap.reindex(data.game_ids).fillna(0).to_numpy()


def knowledge_filter(data, games, players=None, max_minutes=None, age=None, weight_range=None):
    """Mascara (na ordem de data.game_ids) dos jogos que atendem as restricoes informadas.

    Jogos sem a informacao pedida ficam de fora.
    """
    info = games.reindex(data.game_ids)
    keep = pd.Series(True, index=info.index)
    if players:
        keep &= (info["minplayers"] <= players) & (info["maxplayers"] >= players)
    if max_minutes:
        keep &= info["playingtime"] <= max_minutes
    if age:
        keep &= info["minage"] <= age
    if weight_range:
        keep &= info["averageweight"].between(*weight_range)
    return keep.to_numpy()


def find_user(data, name):
    """Devolve o indice do usuario, ou None se o nome nao existir."""
    i = np.searchsorted(data.users, name)
    return int(i) if i < len(data.users) and data.users[i] == name else None


def suggest_users(data, name, limit=5):
    """Nomes parecidos, para quando o usuario digitado nao existe."""
    if not name:
        return []
    matches = data.users[np.char.find(np.char.lower(data.users), name.lower()) >= 0]
    return matches[:limit].tolist()


def pearson_similarities(data, user, min_common=MIN_COMMON):
    """Pearson do usuario contra todos, sobre os jogos em comum e com peso n / (n + 25).

    Devolve (similaridades, jogos em comum). Fica NaN onde nao ha jogos em comum
    suficientes ou onde um dos lados nao varia as notas nesses jogos.
    """
    row = data.by_user[user]
    return pearson_from_ratings(data, row.indices, row.data, min_common, exclude=user)


def pearson_from_ratings(data, games, ratings, min_common=MIN_COMMON, exclude=None):
    """Mesmo calculo de pearson_similarities para notas que nao estao na base."""
    sub = data.by_game[:, games]
    others = sub.indices
    y = sub.data.astype(np.float64)
    x = np.asarray(ratings, dtype=np.float64)[np.repeat(np.arange(sub.shape[1]), np.diff(sub.indptr))]

    size = data.by_user.shape[0]
    n = np.bincount(others, minlength=size).astype(np.float64)
    sum_x = np.bincount(others, weights=x, minlength=size)
    sum_y = np.bincount(others, weights=y, minlength=size)
    sum_xy = np.bincount(others, weights=x * y, minlength=size)
    sum_xx = np.bincount(others, weights=x * x, minlength=size)
    sum_yy = np.bincount(others, weights=y * y, minlength=size)

    sims = np.full(size, np.nan)
    ok = n >= min_common
    if exclude is not None:
        ok[exclude] = False
    cov = sum_xy[ok] - sum_x[ok] * sum_y[ok] / n[ok]
    var_x = sum_xx[ok] - sum_x[ok] ** 2 / n[ok]
    var_y = sum_yy[ok] - sum_y[ok] ** 2 / n[ok]
    varies = (var_x > EPS) & (var_y > EPS)
    values = np.full(len(cov), np.nan)
    values[varies] = cov[varies] / np.sqrt(var_x[varies] * var_y[varies])
    sims[ok] = np.clip(values, -1.0, 1.0) * n[ok] / (n[ok] + SIGNIFICANCE)
    return sims, n


def predict_game(data, mean, std, game, sims, common, n_neighbors=N_NEIGHBORS):
    """Preve a nota de um jogo, sem limitar a escala.

    Devolve (nota, vizinhos usados, similaridade media) ou None. So vizinhos com similaridade positiva entram; empates sao decididos por quem
    tem mais jogos em comum. Jogos com menos de n_neighbors vizinhos ficam sem nota.
    """
    start, end = data.by_game.indptr[game], data.by_game.indptr[game + 1]
    raters = data.by_game.indices[start:end]
    keep = sims[raters] > 0
    if keep.sum() < n_neighbors:
        return None
    raters = raters[keep]
    ratings = data.by_game.data[start:end][keep].astype(np.float64)

    top = np.lexsort((common[raters], sims[raters]))[-n_neighbors:]
    neighbors, weights = raters[top], sims[raters[top]]
    z_scores = (ratings[top] - data.means[neighbors]) / data.stds[neighbors]
    score = mean + std * (weights @ z_scores) / weights.sum()
    return float(score), len(top), float(weights.mean())


def rank_unrated(data, mean, std, rated, sims, common, top=None, penalty=None, allowed=None):
    """Preve os jogos fora de rated e ordena pela pontuacao (nota menos penalty).

    allowed restringe os candidatos; top=None devolve todos. Devolve (tabela,
    jogos com nota, candidatos).
    """
    candidates = np.setdiff1d(np.arange(data.by_game.shape[1]), rated)
    if allowed is not None:
        candidates = candidates[allowed[candidates]]
    rows = []
    for game in candidates:
        result = predict_game(data, mean, std, game, sims, common)
        if result is not None:
            score = result[0] - (penalty[game] if penalty is not None else 0.0)
            rows.append((data.game_ids[game], *result, score))
    table = pd.DataFrame(rows, columns=["game_id", "bruta", "vizinhos", "similaridade", "pontuacao"])
    table = table.sort_values(["pontuacao", "game_id"], ascending=[False, True])
    if top is not None:
        table = table.head(top)
    table["nota"] = table["bruta"].clip(RATING_MIN, RATING_MAX)
    return table, len(rows), len(candidates)


def recommend(data, user, top=10, penalty=None):
    """Avalia todos os jogos que o usuario nao avaliou e devolve os melhores.

    Devolve (tabela, jogos com nota, jogos avaliados, usuarios similares).
    """
    sims, common = pearson_similarities(data, user)
    rated = data.by_user[user].indices
    table, n_scored, n_unrated = rank_unrated(
        data, data.means[user], data.stds[user], rated, sims, common, top, penalty)
    return table, n_scored, n_unrated, int(np.sum(sims > 0))


def fill_with_popular(data, games, table, rated, top, allowed=None):
    """Completa a lista com os jogos de melhor ranking do BGG, que ficam sem nota prevista."""
    table = table.assign(metodo="colaborativo")
    missing = top - len(table)
    if missing <= 0:
        return table
    skip = set(data.game_ids[rated].tolist()) | set(table["game_id"].tolist())
    if allowed is not None:
        skip |= set(data.game_ids[~allowed].tolist())
    ranked = games["board_game_rank"].sort_values().index
    extra = [g for g in ranked if g not in skip][:missing]
    return pd.concat([
        table,
        pd.DataFrame({"game_id": extra, "nota": np.nan, "similaridade": np.nan,
                      "pontuacao": np.nan, "metodo": "popularidade"}),
    ], ignore_index=True)


def empty_table():
    return pd.DataFrame(columns=["game_id", "nota", "similaridade", "pontuacao"])


def recommend_existing_user(data, games, user, top=10):
    """Cenario 1: lista para um usuario antigo, com popularidade quando faltam vizinhos."""
    rated = data.by_user[user].indices
    if data.stds[user] < EPS:
        return fill_with_popular(data, games, empty_table(), rated, top), 0, 0
    table, n_scored, _, n_similar = recommend(data, user, top, product_penalty(data, games))
    return fill_with_popular(data, games, table, rated, top), n_scored, n_similar


def recommend_new_user(data, games, ratings, top=10, filters=None):
    """Cenario 2: lista para quem so tem as notas em ratings ({game_id: nota}).

    filters sao os argumentos de knowledge_filter. Com menos de MIN_COMMON notas
    nenhum vizinho e possivel, entao a lista vem so da popularidade.
    """
    allowed = knowledge_filter(data, games, **filters) if filters else None
    ids = np.array(sorted(ratings), dtype=data.game_ids.dtype)
    rated = np.searchsorted(data.game_ids, ids)
    values = np.array([ratings[g] for g in ids.tolist()], dtype=np.float64)
    mean, std = (values.mean(), values.std()) if len(values) else (0.0, 0.0)
    if len(values) < MIN_COMMON or std < EPS:
        return fill_with_popular(data, games, empty_table(), rated, top, allowed), 0, 0
    sims, common = pearson_from_ratings(data, rated, values)
    table, n_scored, _ = rank_unrated(
        data, mean, std, rated, sims, common, top, product_penalty(data, games), allowed)
    return fill_with_popular(data, games, table, rated, top, allowed), n_scored, int(np.sum(sims > 0))


def predict_new_item(data, games, user, categories, mechanics, ignore_game_id=None, k=N_NEIGHBORS):
    """Cenario 3: preve a nota de um jogo sem avaliacoes para um usuario antigo.

    Sem avaliacoes nao ha vizinhos, entao a mesma media ponderada de z-scores usa
    os k jogos do usuario mais parecidos com o novo (Jaccard de categorias e
    mecanicas). ignore_game_id simula um jogo existente como novo. Devolve None se
    nenhum jogo do usuario tiver algo em comum com o novo.
    """
    new = item_features(categories, mechanics)
    row = data.by_user[user]
    ids = data.game_ids[row.indices]
    values = row.data.astype(np.float64)
    if ignore_game_id is not None:
        keep = ids != ignore_game_id
        ids, values = ids[keep], values[keep]
    if len(values) == 0 or not new:
        return None
    mean, std = values.mean(), values.std()

    feature_sets = games["features"].reindex(ids).tolist()
    sims = np.array([len(new & f) / len(new | f) for f in feature_sets])
    top = np.argsort(-sims, kind="stable")[:k]
    top = top[sims[top] > 0]
    if len(top) == 0:
        return None
    z_scores = (values[top] - mean) / std if std >= EPS else np.zeros(len(top))
    score = mean + std * (sims[top] @ z_scores) / sims[top].sum()
    return {
        "bruta": float(score),
        "nota": float(np.clip(score, RATING_MIN, RATING_MAX)),
        "media_usuario": float(mean),
        "recomendar": bool(score >= mean),
        "parecidos": [
            {"game_id": int(ids[i]), "nota": float(values[i]), "similaridade": float(sims[i])} for i in top
        ],
    }


def show_recommendations(data, games, name, top):
    user = find_user(data, name)
    if user is None:
        print(f"Usuario '{name}' nao encontrado.")
        similar = suggest_users(data, name)
        if similar:
            print("Parecidos:", ", ".join(similar))
        return

    n_rated = data.by_user[user].nnz
    print(f"\n{name}: {n_rated} avaliacoes, media {data.means[user]:.2f}, desvio {data.stds[user]:.2f}")
    if data.stds[user] < EPS:
        print("Este usuario da a mesma nota para tudo: nao ha como calcular Pearson.")
        return

    table, n_scored, n_unrated, n_similar = recommend(data, user, top, product_penalty(data, games))
    print(f"{n_similar:,} usuarios similares; {n_scored:,} de {n_unrated:,} jogos nao avaliados receberam nota.")
    if table.empty:
        print(f"Nenhum jogo tem {N_NEIGHBORS} vizinhos com ao menos {MIN_COMMON} jogos em comum.")
        return

    print(f"\n{'#':>3}  {'pont.':>5}  {'nota':>5}  {'sim.':>5}  {'ano':>5}  jogo")
    for i, row in enumerate(table.itertuples(), 1):
        info = games.loc[row.game_id]
        year = "" if pd.isna(info["yearpublished"]) else int(info["yearpublished"])
        print(f"{i:>3}  {row.pontuacao:>5.2f}  {row.nota:>5.2f}  {row.similaridade:>5.2f}  {year:>5}  {info['name']}")


def main():
    parser = argparse.ArgumentParser(description="Recomenda jogos do BGG para um usuario.")
    parser.add_argument("usuario", nargs="?", help="nome do usuario; sem ele, pergunta no terminal")
    parser.add_argument("--top", type=int, default=10, help="quantas recomendacoes mostrar (padrao: 10)")
    args = parser.parse_args()

    data = load_ratings()
    games = load_game_info()

    if args.usuario:
        show_recommendations(data, games, args.usuario, args.top)
        return

    print(f"{len(data.users):,} usuarios e {len(data.game_ids):,} jogos carregados.")
    while True:
        try:
            name = input("\nUsuario (Enter para sair): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not name:
            break
        show_recommendations(data, games, name, args.top)


if __name__ == "__main__":
    main()
