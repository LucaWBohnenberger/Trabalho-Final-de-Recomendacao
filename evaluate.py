#!/usr/bin/env python3
"""Avaliacao do recomendador (recommend.py).

Secoes:
  A  Erro em avaliacoes escondidas, para as quatro combinacoes de minimo de
     jogos em comum (5 ou 10) e peso n / (n + 25), em usuarios com 20+ e com
     6 a 19 avaliacoes.
  B  Usuario novo: usuarios reais com so 10, 20 ou 30 notas visiveis.
  C  Item novo: um jogo do usuario tratado como recem-adicionado.
  D  Listas completas para usuarios sorteados: quem fica sem recomendacao,
     cobertura do catalogo e quais jogos se repetem.
  E  Qualidade da lista com o escore de produto: esconde um jogo de que o
     usuario gostou (nota 8+) e mede em que posicao ele aparece, para cada peso
     do desconto, e quanto os mesmos jogos se repetem entre usuarios.

Uso:
    python3 evaluate.py                    # todas as secoes
    python3 evaluate.py --secoes A,D --workers 4

O progresso sai no terminal. Os resultados vao para clean/avaliacao.json, que e
salvo ao fim de cada secao: uma secao que falha nao derruba as outras, e um caso
que falha so e contado e registrado.
"""
import argparse
import collections
import functools
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse

import recommend as rec

OUT = rec.BASE / "clean" / "avaliacao.json"
SECTIONS = "ABCDE"
VARIANTS = [(5, 0), (5, 25), (10, 0), (10, 25)]
LIST_VARIANTS = [(5, 25), (10, 25)]
NEW_USER_VISIBLE = (10, 20, 30)
BANDS = [(1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, None)]
PRODUCT_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
LIKED = 8.0
CUTOFFS = (10, 50)
MAX_EXAMPLES = 5

# As secoes aplicam o minimo e o peso por conta propria (veja variant_sims).
rec.SIGNIFICANCE = 0

# Preenchidos antes de criar os processos; com fork, cada um herda uma copia.
DATA = None
GAMES = None
HOLDOUT = None


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def fmt_seconds(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s"


class Progress:
    def __init__(self, label, total, every=20.0):
        self.label, self.total, self.every = label, total, every
        self.done = self.failed = 0
        self.start = self.last = time.time()

    def step(self, failed=False):
        self.done += 1
        self.failed += failed
        now = time.time()
        if self.done < self.total and now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.start
        remaining = (self.total - self.done) * elapsed / self.done
        errors = f", {self.failed} com erro" if self.failed else ""
        log(f"[{self.label}] {self.done}/{self.total} ({self.done / self.total:.0%}){errors}"
            f" - {fmt_seconds(elapsed)} decorridos, faltam ~{fmt_seconds(remaining)}")


def guarded(func, task):
    try:
        return True, func(task)
    except Exception as err:
        return False, f"{type(err).__name__}: {err}"


def run_parallel(label, func, tasks, workers):
    """Roda func em cada tarefa. Devolve (resultados, erros); um caso com erro nao para os outros."""
    results, errors = [], collections.Counter()
    progress = Progress(label, len(tasks))
    with mp.get_context("fork").Pool(workers) as pool:
        for ok, value in pool.imap_unordered(functools.partial(guarded, func), tasks, chunksize=4):
            if ok:
                results.append(value)
            else:
                errors[value] += 1
            progress.step(failed=not ok)
    return results, {"total": sum(errors.values()), "exemplos": dict(errors.most_common(MAX_EXAMPLES))}


def metrics(pred, truth):
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    if len(pred) == 0:
        return None
    err = pred - truth
    return {"n": len(pred), "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "mae": round(float(np.mean(np.abs(err))), 4)}


def ratio(part, total):
    return round(part / total, 4) if total else None


def variant_name(min_common, weight):
    return f"min{min_common}_{'peso' if weight else 'sem_peso'}"


def variant_sims(raw, common, min_common, weight):
    """Aplica o minimo de jogos em comum e o peso n / (n + weight) ao Pearson puro."""
    sims = np.where(common >= min_common, raw, np.nan)
    return sims * common / (common + weight) if weight else sims


def raw_pearson(data, games, ratings, exclude):
    return rec.pearson_from_ratings(data, games, ratings, min_common=min(m for m, _ in VARIANTS), exclude=exclude)


def user_stats(by_user):
    rows = np.repeat(np.arange(by_user.shape[0]), np.diff(by_user.indptr))
    values = by_user.data.astype(np.float64)
    counts = np.bincount(rows, minlength=by_user.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.bincount(rows, weights=values, minlength=by_user.shape[0]) / counts
        stds = np.sqrt(np.bincount(rows, weights=(values - means[rows]) ** 2, minlength=by_user.shape[0]) / counts)
    return means, stds


def game_mean(data, game, hidden=None):
    """Media do jogo, descontando a nota hidden se ela ainda estiver na matriz; None se nao sobra ninguem."""
    start, end = data.by_game.indptr[game], data.by_game.indptr[game + 1]
    total, count = float(data.by_game.data[start:end].sum()), end - start
    if hidden is not None:
        total, count = total - hidden, count - 1
    return total / count if count > 0 else None


def cutoff(data, game, sims, common):
    """(10o e 11o vizinhos empatados?, mediana de jogos em comum dos 10 escolhidos)."""
    start, end = data.by_game.indptr[game], data.by_game.indptr[game + 1]
    raters = data.by_game.indices[start:end]
    raters = raters[sims[raters] > 0]
    order = raters[np.lexsort((common[raters], sims[raters]))]
    tenth = order[-rec.N_NEIGHBORS]
    tie = len(order) > rec.N_NEIGHBORS and sims[order[-rec.N_NEIGHBORS - 1]] == sims[tenth] \
        and common[order[-rec.N_NEIGHBORS - 1]] == common[tenth]
    return bool(tie), float(np.median(common[order[-rec.N_NEIGHBORS:]]))


# --- A: avaliacoes escondidas -------------------------------------------------

def build_holdout(data, per_group, seed):
    """Esconde 1 nota de cada usuario sorteado e recalcula as matrizes e as medias."""
    by_user = data.by_user
    counts = np.diff(by_user.indptr)
    groups = {"20+": np.flatnonzero(counts >= 20), "6-19": np.flatnonzero((counts >= 6) & (counts < 20))}
    rng = np.random.default_rng(seed)
    keep = np.ones(by_user.nnz, bool)
    tests = []
    for group, pool in groups.items():
        for user in rng.choice(pool, min(per_group, len(pool)), replace=False):
            pos = rng.integers(by_user.indptr[user], by_user.indptr[user + 1])
            keep[pos] = False
            tests.append((int(user), int(by_user.indices[pos]), float(by_user.data[pos]), group))

    return without(data, keep), tests


def without(data, keep):
    """Copia de data so com as notas em que keep e verdadeiro, com medias e desvios refeitos."""
    by_user = data.by_user
    rows = np.repeat(np.arange(by_user.shape[0]), np.diff(by_user.indptr))
    hidden = sparse.csr_matrix((by_user.data[keep], (rows[keep], by_user.indices[keep])), shape=by_user.shape)
    means, stds = user_stats(hidden)
    return rec.Ratings(hidden, hidden.tocsc(), data.users, data.game_ids, means, stds)


def holdout_case(i):
    data, (user, game, truth, group) = HOLDOUT["data"], HOLDOUT["tests"][i]
    case = {"grupo": group, "real": truth, "media_usuario": float(data.means[user]),
            "media_jogo": game_mean(data, game)}
    mean, std = data.means[user], data.stds[user]
    raw = common = None
    if std >= rec.EPS:
        row = data.by_user[user]
        raw, common = raw_pearson(data, row.indices, row.data, exclude=user)
    for min_common, weight in VARIANTS:
        name = variant_name(min_common, weight)
        case[name] = None
        if raw is None:
            continue
        sims = variant_sims(raw, common, min_common, weight)
        result = rec.predict_game(data, mean, std, game, sims, common)
        if result is not None:
            tie, median_common = cutoff(data, game, sims, common)
            case[name] = {"nota": float(np.clip(result[0], rec.RATING_MIN, rec.RATING_MAX)),
                          "empate": tie, "comum": median_common}
    return case


def summarize_holdout(cases):
    names = [variant_name(*v) for v in VARIANTS]
    summary = {}
    for group in sorted({c["grupo"] for c in cases}):
        group_cases = [c for c in cases if c["grupo"] == group]
        shared = [c for c in group_cases if c["media_jogo"] is not None and all(c[n] for n in names)]
        truth = [c["real"] for c in shared]
        out = {
            "casos": len(group_cases),
            "conjunto_comum": len(shared),
            "media_usuario": metrics([c["media_usuario"] for c in shared], truth),
            "media_jogo": metrics([c["media_jogo"] for c in shared], truth),
        }
        for name in names:
            covered = [c for c in group_cases if c[name]]
            out[name] = {
                "cobertura": ratio(len(covered), len(group_cases)),
                "erro_proprio": metrics([c[name]["nota"] for c in covered], [c["real"] for c in covered]),
                "erro_comum": metrics([c[name]["nota"] for c in shared], truth),
                "empates_no_corte": ratio(sum(c[name]["empate"] for c in covered), len(covered)),
                "mediana_jogos_em_comum": float(np.median([c[name]["comum"] for c in covered])) if covered else None,
            }
        summary[group] = out
    return summary


def section_a(workers, per_group=3000, seed=7):
    global HOLDOUT
    log("Escondendo uma nota por usuario e recalculando as medias...")
    data, tests = build_holdout(DATA, per_group, seed)
    HOLDOUT = {"data": data, "tests": tests}
    cases, errors = run_parallel("A", holdout_case, list(range(len(tests))), workers)
    HOLDOUT = None
    return {"resultado": summarize_holdout(cases), "erros": errors}


# --- B: usuario novo -----------------------------------------------------------

def new_user_case(task):
    user, seed = task
    row = DATA.by_user[user]
    order = np.random.default_rng(seed).permutation(row.nnz)
    game, truth = row.indices[order[0]], float(row.data[order[0]])
    case = {"real": truth, "media_jogo": game_mean(DATA, game, hidden=truth)}
    for visible in NEW_USER_VISIBLE:
        shown = order[1:visible + 1]
        games, values = row.indices[shown], row.data[shown].astype(np.float64)
        mean, std = values.mean(), values.std()
        entry = {"media_visivel": float(mean)}
        raw = common = None
        if std >= rec.EPS:
            raw, common = raw_pearson(DATA, games, values, exclude=user)
        for min_common, weight in LIST_VARIANTS:
            name = variant_name(min_common, weight)
            entry[name] = None
            if raw is not None:
                sims = variant_sims(raw, common, min_common, weight)
                result = rec.predict_game(DATA, mean, std, game, sims, common)
                if result is not None:
                    entry[name] = float(np.clip(result[0], rec.RATING_MIN, rec.RATING_MAX))
        case[visible] = entry
    return case


def section_b(workers, n_users=2000, min_ratings=40, seed=3):
    counts = np.diff(DATA.by_user.indptr)
    users = np.random.default_rng(seed).choice(np.flatnonzero(counts >= min_ratings), n_users, replace=False)
    cases, errors = run_parallel("B", new_user_case, [(int(u), i) for i, u in enumerate(users)], workers)
    summary = {}
    for visible in NEW_USER_VISIBLE:
        out = {}
        for min_common, weight in LIST_VARIANTS:
            name = variant_name(min_common, weight)
            covered = [c for c in cases if c[visible][name] is not None and c["media_jogo"] is not None]
            truth = [c["real"] for c in covered]
            out[name] = {
                "cobertura": ratio(len(covered), len(cases)),
                "modelo": metrics([c[visible][name] for c in covered], truth),
                "media_visivel": metrics([c[visible]["media_visivel"] for c in covered], truth),
                "media_jogo": metrics([c["media_jogo"] for c in covered], truth),
            }
        summary[f"{visible}_notas"] = out
    return {"resultado": summary, "usuarios": len(cases), "erros": errors}


# --- C: item novo --------------------------------------------------------------

def new_item_case(task):
    user, seed = task
    row = DATA.by_user[user]
    pos = np.random.default_rng(seed).integers(row.nnz)
    game, truth = row.indices[pos], float(row.data[pos])
    game_id = int(DATA.game_ids[game])
    result = rec.predict_new_item(
        DATA, GAMES, user,
        GAMES.at[game_id, "boardgamecategory"], GAMES.at[game_id, "boardgamemechanic"],
        ignore_game_id=game_id,
    )
    return {
        "real": truth,
        "media_usuario": float(np.delete(row.data, pos).astype(np.float64).mean()),
        "media_jogo": game_mean(DATA, game, hidden=truth),
        "nota": None if result is None else result["nota"],
        "recomendar": None if result is None else result["recomendar"],
    }


def section_c(workers, n_users=3000, min_ratings=20, seed=5):
    counts = np.diff(DATA.by_user.indptr)
    users = np.random.default_rng(seed).choice(np.flatnonzero(counts >= min_ratings), n_users, replace=False)
    cases, errors = run_parallel("C", new_item_case, [(int(u), i) for i, u in enumerate(users)], workers)
    covered = [c for c in cases if c["nota"] is not None]
    truth = np.array([c["real"] for c in covered])
    with_game_mean = [c for c in covered if c["media_jogo"] is not None]
    recommended = np.array([c["recomendar"] for c in covered], bool)
    liked = truth >= np.array([c["media_usuario"] for c in covered])
    predicted_gap = np.array([c["nota"] - c["media_usuario"] for c in covered])
    real_gap = truth - np.array([c["media_usuario"] for c in covered])
    return {
        "resultado": {
            "cobertura": ratio(len(covered), len(cases)),
            "modelo": metrics([c["nota"] for c in covered], truth),
            "media_usuario": metrics([c["media_usuario"] for c in covered], truth),
            "media_jogo_referencia": metrics([c["media_jogo"] for c in with_game_mean],
                                             [c["real"] for c in with_game_mean]),
            "fracao_recomendada": ratio(int(recommended.sum()), len(covered)),
            "gostou_se_recomendado": ratio(int(liked[recommended].sum()), int(recommended.sum())),
            "gostou_se_nao_recomendado": ratio(int(liked[~recommended].sum()), int((~recommended).sum())),
            "correlacao": round(float(np.corrcoef(predicted_gap, real_gap)[0, 1]), 4) if len(covered) > 2 else None,
        },
        "usuarios": len(cases),
        "erros": errors,
    }


# --- D: listas completas -------------------------------------------------------

def list_case(user):
    row = DATA.by_user[user]
    case = {"avaliacoes": row.nnz, "constante": bool(DATA.stds[user] < rec.EPS)}
    if case["constante"]:
        return case
    raw, common = raw_pearson(DATA, row.indices, row.data, exclude=user)
    for min_common, weight in LIST_VARIANTS:
        sims = variant_sims(raw, common, min_common, weight)
        table, scored, unrated = rec.rank_unrated(
            DATA, DATA.means[user], DATA.stds[user], row.indices, sims, common, rec.N_NEIGHBORS)
        case[variant_name(min_common, weight)] = {
            "com_nota": scored, "nao_avaliados": unrated, "top": table["game_id"].astype(int).tolist()}
    return case


def section_d(workers, n_users=400, seed=11):
    users = np.random.default_rng(seed).choice(DATA.by_user.shape[0], n_users, replace=False)
    cases, errors = run_parallel("D", list_case, [int(u) for u in users], workers)
    reviews = dict(zip(DATA.game_ids.tolist(), np.diff(DATA.by_game.indptr).tolist()))
    summary = {"usuarios": len(cases), "notas_todas_iguais": sum(c["constante"] for c in cases),
               "mediana_avaliacoes_catalogo": float(np.median(list(reviews.values())))}
    for min_common, weight in LIST_VARIANTS:
        name = variant_name(min_common, weight)
        bands = {}
        for low, high in BANDS:
            band = [c for c in cases if c["avaliacoes"] >= low and (high is None or c["avaliacoes"] <= high)]
            usable = [c[name] for c in band if not c["constante"]]
            with_list = [u for u in usable if u["top"]]
            bands[f"{low}-{high or 'mais'}"] = {
                "usuarios": len(band),
                "sem_recomendacao": len(band) - len(with_list),
                "mediana_cobertura": round(float(np.median([ratio(u["com_nota"], u["nao_avaliados"]) or 0
                                                            for u in with_list])), 4) if with_list else None,
            }
        tops = [c[name]["top"] for c in cases if not c["constante"] and c[name]["top"]]
        counter = collections.Counter(g for top in tops for g in top)
        summary[name] = {
            "faixas": bands,
            "usuarios_com_lista": len(tops),
            "jogos_distintos_nos_top10": len(counter),
            "mais_repetidos": [
                {"jogo": GAMES.at[g, "name"], "listas": n, "fracao": ratio(n, len(tops)), "avaliacoes": reviews[g]}
                for g, n in counter.most_common(8)
            ],
            "mediana_avaliacoes_recomendados": float(np.median([reviews[g] for top in tops for g in top]))
            if tops else None,
        }
    return {"resultado": summary, "erros": errors}


# --- E: qualidade da lista e escore de produto --------------------------------

def position(order, game):
    """Posicao (1 = primeiro) de game em order, ou None se ele nao esta na lista."""
    found = np.flatnonzero(order == game)
    return int(found[0]) + 1 if len(found) else None


def list_quality_case(i):
    data, (user, game) = HOLDOUT["data"], HOLDOUT["tests"][i]
    row = data.by_user[user]
    raw, common = raw_pearson(data, row.indices, row.data, exclude=user)
    sims = variant_sims(raw, common, rec.MIN_COMMON, 25)
    table, _, _ = rec.rank_unrated(data, data.means[user], data.stds[user], row.indices, sims, common)
    scored = np.searchsorted(data.game_ids, table["game_id"].to_numpy())
    case = {"previsto": bool(np.any(scored == game))}
    for weight in PRODUCT_WEIGHTS:
        score = table["bruta"].to_numpy() - weight * HOLDOUT["gap"][scored]
        order = scored[np.lexsort((scored, -score))]
        case[weight] = {"posicao": position(order, game), "top": data.game_ids[order[:10]].tolist()}
    unrated = HOLDOUT["by_rank"][~np.isin(HOLDOUT["by_rank"], row.indices)]
    case["popularidade"] = {"posicao": position(unrated, game), "top": data.game_ids[unrated[:10]].tolist()}
    return case


def summarize_lists(cases, key, reviews):
    positions = [c[key]["posicao"] for c in cases]
    found = [p for p in positions if p is not None]
    tops = [c[key]["top"] for c in cases]
    counter = collections.Counter(g for top in tops for g in top)
    out = {f"acerto@{k}": ratio(sum(p <= k for p in found), len(cases)) for k in CUTOFFS}
    out.update({
        "mrr": round(float(np.mean([1 / p if p else 0.0 for p in positions])), 5) if cases else None,
        "mediana_posicao": float(np.median(found)) if found else None,
        "jogos_distintos_nos_top10": len(counter),
        "mais_repetidos": [
            {"jogo": GAMES.at[g, "name"], "fracao": ratio(n, len(tops)), "avaliacoes": reviews[g]}
            for g, n in counter.most_common(5)
        ],
        "mediana_avaliacoes_recomendados": float(np.median([reviews[g] for top in tops for g in top]))
        if tops else None,
    })
    return out


def section_e(workers, n_users=1000, min_ratings=20, seed=13):
    global HOLDOUT
    by_user = DATA.by_user
    counts = np.diff(by_user.indptr)
    rng = np.random.default_rng(seed)
    keep = np.ones(by_user.nnz, bool)
    tests = []
    for user in rng.permutation(np.flatnonzero(counts >= min_ratings)):
        start, end = by_user.indptr[user], by_user.indptr[user + 1]
        liked = start + np.flatnonzero(by_user.data[start:end] >= LIKED)
        if DATA.stds[user] < rec.EPS or len(liked) == 0:
            continue
        pos = rng.choice(liked)
        keep[pos] = False
        tests.append((int(user), int(by_user.indices[pos])))
        if len(tests) == n_users:
            break

    log("Escondendo um jogo de que cada usuario gostou...")
    ranks = GAMES["board_game_rank"].reindex(DATA.game_ids).to_numpy()
    HOLDOUT = {
        "data": without(DATA, keep),
        "tests": tests,
        "gap": rec.product_penalty(DATA, GAMES, weight=1.0),
        "by_rank": np.argsort(ranks, kind="stable"),
    }
    cases, errors = run_parallel("E", list_quality_case, list(range(len(tests))), workers)
    HOLDOUT = None
    reviews = dict(zip(DATA.game_ids.tolist(), np.diff(DATA.by_game.indptr).tolist()))
    summary = {"usuarios": len(cases), "jogo_escondido_previsto": ratio(sum(c["previsto"] for c in cases), len(cases))}
    for weight in PRODUCT_WEIGHTS:
        summary[f"peso_{weight}"] = summarize_lists(cases, weight, reviews)
    summary["so_popularidade"] = summarize_lists(cases, "popularidade", reviews)
    return {"resultado": summary, "erros": errors}


# --- execucao ------------------------------------------------------------------

def to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} nao e serializavel")


def save(results, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=to_json), encoding="utf-8")
    os.replace(tmp, path)


def main():
    global DATA, GAMES
    parser = argparse.ArgumentParser(description="Avaliacao do recomendador.")
    parser.add_argument("--secoes", default=",".join(SECTIONS), help="ex.: A,C (padrao: todas)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--saida", default=str(OUT), help=f"arquivo JSON (padrao: {OUT})")
    args = parser.parse_args()
    chosen = [s.strip().upper() for s in args.secoes.split(",") if s.strip()]
    path = Path(args.saida)

    runners = {"A": section_a, "B": section_b, "C": section_c, "D": section_d, "E": section_e}
    log(f"Carregando os dados (secoes {', '.join(chosen)}, {args.workers} processos)...")
    DATA, GAMES = rec.load_ratings(), rec.load_game_info()
    # Rodar so algumas secoes atualiza essas e preserva as outras ja salvas.
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        results = {}
    try:
        for name in chosen:
            if name not in runners:
                log(f"Secao {name} nao existe; ignorada.")
                continue
            log(f"=== Secao {name} ===")
            start = time.time()
            try:
                results[name] = {"status": "ok", **runners[name](args.workers)}
                log(f"Secao {name} concluida em {fmt_seconds(time.time() - start)}.")
            except Exception:
                results[name] = {"status": "falhou", "erro": traceback.format_exc()}
                log(f"Secao {name} falhou; seguindo para a proxima.\n{results[name]['erro']}")
            results[name]["segundos"] = round(time.time() - start, 1)
            save(results, path)
    except KeyboardInterrupt:
        log("Interrompido; salvando o que ja terminou.")
        save(results, path)
    log(f"Resultados em {path}")


if __name__ == "__main__":
    main()
