#!/usr/bin/env python3
"""Interface web para testar o recomendador (recommend.py).

Uso:    python3 src/app.py [--porta 8000]
Depois: http://127.0.0.1:8000
"""
import argparse
import json
import math
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import recommend as rec

MIN_RATINGS_RANDOM_USER = 20
SEARCH_LIMIT = 10
USER_SEARCH_LIMIT = 8
POPULAR_FORM_SIZE = 20

PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recomendador BGG</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f5f6f8; --card: #ffffff; --text: #1f2430; --muted: #646b7a;
    --line: #dfe2e8; --accent: #2f5fd0; --accent-text: #ffffff; --error: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161b; --card: #1e2128; --text: #e8eaef; --muted: #9aa1ae;
      --line: #333843; --accent: #7c9ff0; --accent-text: #10131a; --error: #f2857d;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, sans-serif; line-height: 1.45; color: var(--text); background: var(--bg);
    max-width: 900px; margin: 0 auto; padding: 24px 16px;
  }
  h1 { font-size: 1.6rem; margin: 0 0 16px; }
  h2 { font-size: 1.15rem; margin: 0 0 12px; }
  h3 { font-size: 1rem; margin: 16px 0 4px; }
  p { color: var(--muted); }
  nav { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid var(--line); }
  nav button {
    background: none; border: 0; border-bottom: 2px solid transparent; border-radius: 0;
    color: var(--muted); padding: 10px 14px;
  }
  nav button:hover:not(:disabled) { background: none; color: var(--text); }
  nav button.on { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
  section {
    display: none; margin-top: 16px; padding: 20px;
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  }
  section.on { display: block; }
  label { display: block; margin: 12px 0 4px; font-size: 0.9rem; font-weight: 600; }
  input, select, button { font: inherit; color: inherit; }
  input, select {
    padding: 7px 10px; background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  }
  input:not([type=number]) { width: min(100%, 320px); }
  input[type=number] { width: 72px; }
  select[multiple] { width: 100%; }
  button {
    padding: 7px 12px; background: var(--card); border: 1px solid var(--line); border-radius: 6px; cursor: pointer;
  }
  button:hover:not(:disabled) { background: var(--bg); }
  button:disabled { opacity: 0.6; cursor: wait; }
  #go1, #go2, #go3 { background: var(--accent); border-color: var(--accent); color: var(--accent-text); font-weight: 600; }
  #go1:hover:not(:disabled), #go2:hover:not(:disabled), #go3:hover:not(:disabled) { opacity: 0.88; background: var(--accent); }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 0.93rem; }
  th, td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; }
  th { color: var(--muted); font-size: 0.8rem; font-weight: 600; }
  .erro { color: var(--error); }
  .result { margin-top: 16px; overflow-x: auto; }
  .filters { display: flex; flex-wrap: wrap; gap: 0 16px; }
  .filters label { font-weight: normal; }
  .filters.off { opacity: 0.45; }
  .toggle { display: inline-flex; align-items: center; gap: 8px; margin: 16px 0 4px; font-size: 1rem; cursor: pointer; }
  .toggle input { width: 18px; height: 18px; margin: 0; accent-color: var(--accent); cursor: pointer; }
  .hits { margin-top: 4px; }
  .hits button { display: block; width: min(100%, 420px); margin: 2px 0; text-align: left; }
  .ac { position: relative; display: inline-block; width: min(100%, 320px); vertical-align: top; }
  .ac input { width: 100%; }
  .ac .hits {
    position: absolute; top: calc(100% + 4px); left: 0; right: 0; z-index: 10; margin: 0; padding: 4px;
    background: var(--card); border: 1px solid var(--line); border-radius: 8px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
  }
  .ac .hits:empty { display: none; }
  .ac .hits button { width: 100%; margin: 0; border: 0; display: flex; justify-content: space-between; gap: 12px; }
  .ac .hits button.sel, .ac .hits button:hover { background: var(--bg); }
  .ac .hits small { color: var(--muted); }
  .perfil { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 12px; }
  .perfil div { background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px; }
  .perfil b { display: block; font-size: 1.15rem; }
  .perfil span { color: var(--muted); font-size: 0.8rem; }
  .aviso { margin: 0 0 8px; }
  /* Jogos avaliados pelo usuario da secao 3: na lateral esquerda quando cabe, senao no topo. */
  #item { position: relative; }
  #rated3 { margin: 0 0 16px; padding: 14px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
  #rated3:empty { display: none; }
  #rated3 h3 { margin: 0 0 2px; }
  #rated3 p { margin: 0 0 6px; font-size: 0.85rem; }
  #rated3 .scroll { max-height: 240px; overflow-y: auto; }
  #rated3 table { margin: 0; font-size: 0.85rem; }
  #rated3 th { position: sticky; top: 0; background: var(--card); }
  #rated3 td, #rated3 th { padding: 4px 6px; }
  #rated3 tr.ignorado td { color: var(--muted); text-decoration: line-through; }
  @media (min-width: 1240px) {
    body:has(#item.on) { max-width: 1220px; padding-left: 320px; }
    #rated3 { position: absolute; top: 0; right: calc(100% + 24px); width: 280px; margin: 0; }
    #rated3 .scroll { max-height: 70vh; }
  }
</style>
</head>
<body>
<h1>Recomendador BGG</h1>
<nav>
  <button data-tab="antigo" class="on">1. Usuário antigo</button>
  <button data-tab="novo">2. Usuário novo</button>
  <button data-tab="item">3. Item novo</button>
</nav>

<section id="antigo" class="on">
  <h2>Lista para um usuário antigo</h2>
  <label for="u1">Usuário</label>
  <div class="ac"><input id="u1" value="Torsten" autocomplete="off"><div class="hits" id="hu1"></div></div>
  <button data-random="u1">Usuário aleatório</button>
  <button id="go1">Recomendar</button>
  <div class="result" id="r1"></div>
</section>

<section id="novo">
  <h2>Lista para um usuário novo</h2>
  <p>Dê notas de 1 a 10. Com menos de 5 notas, a lista vem só da popularidade; a partir de 20, a previsão fica mais confiável.</p>
  <button id="fill">Preencher 12 notas aleatórias</button>
  <button id="clear">Limpar</button>
  <label for="s2">Adicionar outro jogo</label>
  <input id="s2" placeholder="Nome do jogo">
  <div class="hits" id="h2"></div>
  <table><thead><tr><th>Jogo</th><th>Nota</th></tr></thead><tbody id="form2"></tbody></table>
  <label class="toggle"><input type="checkbox" id="use-filters"> Usar filtros</label>
  <div class="filters off" id="filters">
    <div><label for="f-players">Jogadores</label><input id="f-players" type="number" min="1" max="20"></div>
    <div><label for="f-minutes">Duração máxima (min)</label><input id="f-minutes" type="number" min="5" step="5"></div>
    <div><label for="f-age">Idade do mais novo</label><input id="f-age" type="number" min="1" max="21"></div>
    <div><label for="f-wmin">Complexidade de</label><input id="f-wmin" type="number" min="1" max="5" step="0.5"></div>
    <div><label for="f-wmax">até</label><input id="f-wmax" type="number" min="1" max="5" step="0.5"></div>
  </div>
  <p><button id="go2">Recomendar</button></p>
  <div class="result" id="r2"></div>
</section>

<section id="item">
  <aside id="rated3"></aside>
  <h2>Item novo para um usuário antigo</h2>
  <label for="u3">Usuário</label>
  <div class="ac"><input id="u3" value="Torsten" autocomplete="off"><div class="hits" id="hu3"></div></div>
  <button data-random="u3">Usuário aleatório</button>
  <label for="s3">Simular com um jogo existente (ignora as avaliações dele)</label>
  <input id="s3" placeholder="Nome do jogo">
  <div class="hits" id="h3"></div>
  <label for="name3">Nome do item novo</label>
  <input id="name3" value="Meu jogo novo">
  <label for="cat3">Categorias (Ctrl para escolher várias)</label>
  <select id="cat3" multiple size="6"></select>
  <label for="mec3">Mecânicas (Ctrl para escolher várias)</label>
  <select id="mec3" multiple size="6"></select>
  <div class="filters" id="num3">
    <div><label for="i-weight">Complexidade (1 a 5)</label><input id="i-weight" type="number" min="1" max="5" step="0.1"></div>
    <div><label for="i-minutes">Duração (min)</label><input id="i-minutes" type="number" min="5" step="5"></div>
    <div><label for="i-pmin">Jogadores de</label><input id="i-pmin" type="number" min="1" max="20"></div>
    <div><label for="i-pmax">até</label><input id="i-pmax" type="number" min="1" max="100"></div>
    <div><label for="i-age">Idade mínima</label><input id="i-age" type="number" min="1" max="21"></div>
  </div>
  <p>Campos em branco não entram na semelhança.</p>
  <p><button id="go3">Prever</button></p>
  <div class="result" id="r3"></div>
</section>

<script>
const $ = id => document.getElementById(id);
let ignoreGame = null;

async function api(path, body) {
  const res = await fetch(path, body ? {method: "POST", body: JSON.stringify(body)} : {});
  const data = await res.json();
  if (!res.ok) throw new Error(data.erro);
  return data;
}

function el(tag, text) {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function makeTable(headers, rows) {
  const t = el("table");
  const head = t.createTHead().insertRow();
  headers.forEach(h => head.appendChild(el("th", h)));
  const body = t.createTBody();
  rows.forEach(r => {
    const tr = body.insertRow();
    r.forEach(c => tr.appendChild(el("td", c)));
  });
  return t;
}

const fmt = v => v === null ? "-" : v.toLocaleString("pt-BR", {minimumFractionDigits: 2, maximumFractionDigits: 2});
const fmtInt = v => v.toLocaleString("pt-BR");

function profile(perfil) {
  const box = el("div");
  box.className = "perfil";
  [["Avaliações", fmtInt(perfil.avaliacoes)], ["Média", fmt(perfil.media)],
   ["Desvio", fmt(perfil.desvio)], ["Usuários similares", fmtInt(perfil.similares)]].forEach(([label, value]) => {
    const card = el("div");
    card.append(el("b", value), el("span", label));
    box.appendChild(card);
  });
  return box;
}

function showList(box, data) {
  const head = [];
  if (data.perfil) head.push(profile(data.perfil));
  if (data.resumo) {
    const p = el("p", data.resumo);
    p.className = "aviso";
    head.push(p);
  }
  box.replaceChildren(
    ...head,
    makeTable(["#", "Jogo", "Ano", "Pontuação", "Nota", "Similaridade", "Método"],
      data.itens.map((x, i) => [
        i + 1, x.nome, x.ano, fmt(x.pontuacao), fmt(x.nota), fmt(x.similaridade), x.metodo]))
  );
}

async function run(box, button, action) {
  box.replaceChildren(el("p", "Calculando..."));
  button.disabled = true;
  try {
    await action();
  } catch (err) {
    const p = el("p", err.message);
    p.className = "erro";
    box.replaceChildren(p);
  }
  button.disabled = false;
}

function setupSearch(input, hits, onPick) {
  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      const found = q.length < 2 ? [] : await api("/api/search?q=" + encodeURIComponent(q));
      hits.replaceChildren(...found.map(g => {
        const b = el("button", g.ano ? `${g.nome} (${g.ano})` : g.nome);
        b.onclick = () => { onPick(g); input.value = ""; hits.replaceChildren(); };
        return b;
      }));
    }, 200);
  });
}

function setupUserSearch(input, hits) {
  let timer, pos = -1;
  const buttons = () => [...hits.querySelectorAll("button")];
  const close = () => { hits.replaceChildren(); pos = -1; };
  const pick = name => { input.value = name; close(); input.dispatchEvent(new Event("change")); };
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      const found = q ? await api("/api/usuarios?q=" + encodeURIComponent(q)) : [];
      if (input.value.trim() !== q) return;
      pos = -1;
      hits.replaceChildren(...found.map(u => {
        const b = el("button");
        b.type = "button";
        b.append(el("span", u.usuario), el("small", `${fmtInt(u.avaliacoes)} avaliações`));
        b.onmousedown = e => { e.preventDefault(); pick(u.usuario); };
        return b;
      }));
    }, 150);
  });
  input.addEventListener("keydown", e => {
    const list = buttons();
    if (!list.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      pos = (pos + (e.key === "ArrowDown" ? 1 : -1) + list.length) % list.length;
      list.forEach((b, i) => b.classList.toggle("sel", i === pos));
    } else if (e.key === "Enter" && pos >= 0) {
      e.preventDefault();
      pick(list[pos].firstChild.textContent);
    } else if (e.key === "Escape") {
      close();
    }
  });
  input.addEventListener("blur", close);
}

function addRating(game, value) {
  if (document.querySelector(`#form2 input[data-id="${game.id}"]`)) return;
  const tr = $("form2").insertRow();
  tr.insertCell().textContent = game.nome;
  const input = el("input");
  input.type = "number"; input.min = 1; input.max = 10; input.step = 0.5;
  input.dataset.id = game.id;
  if (value) input.value = value;
  tr.insertCell().appendChild(input);
}

document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button, section").forEach(e => e.classList.remove("on"));
  b.classList.add("on");
  $(b.dataset.tab).classList.add("on");
});

document.querySelectorAll("[data-random]").forEach(b => b.onclick = async () => {
  $(b.dataset.random).value = (await api("/api/random-user")).usuario;
  $(b.dataset.random).dispatchEvent(new Event("change"));
});

setupUserSearch($("u1"), $("hu1"));
setupUserSearch($("u3"), $("hu3"));

// Lista lateral com as notas do usuario da secao 3; o jogo simulado aparece riscado.
let ratedTimer, ratedGames = null;
function drawRated(data) {
  if (!data) return $("rated3").replaceChildren();
  const table = makeTable(["Jogo", "Nota"], data.jogos.map(g => [g.nome, g.nota.toLocaleString("pt-BR", {minimumFractionDigits: 1})]));
  data.jogos.forEach((g, i) => { if (g.id === ignoreGame) table.rows[i + 1].className = "ignorado"; });
  const scroll = el("div");
  scroll.className = "scroll";
  scroll.appendChild(table);
  $("rated3").replaceChildren(el("h3", `Jogos de ${data.usuario}`),
    el("p", `${fmtInt(data.jogos.length)} avaliações, média ${fmt(data.media)}`), scroll);
}
function loadRated() {
  clearTimeout(ratedTimer);
  ratedTimer = setTimeout(async () => {
    const name = $("u3").value.trim();
    let data = null;
    try { if (name) data = await api("/api/avaliacoes?usuario=" + encodeURIComponent(name)); } catch (err) {}
    if ($("u3").value.trim() !== name) return;
    ratedGames = data;
    drawRated(data);
  }, 300);
}
$("u3").addEventListener("input", loadRated);
$("u3").addEventListener("change", loadRated);
loadRated();

$("go1").onclick = () => run($("r1"), $("go1"), async () => {
  showList($("r1"), await api("/api/antigo", {usuario: $("u1").value.trim()}));
});

$("fill").onclick = () => {
  const inputs = [...document.querySelectorAll("#form2 input")];
  inputs.forEach(i => i.value = "");
  inputs.sort(() => Math.random() - 0.5).slice(0, 12)
    .forEach(i => i.value = 1 + Math.floor(Math.random() * 10));
};
$("clear").onclick = () => document.querySelectorAll("#form2 input").forEach(i => i.value = "");

// Com a caixa desmarcada, os campos ficam bloqueados e nenhum filtro e enviado.
function syncFilters() {
  const on = $("use-filters").checked;
  $("filters").classList.toggle("off", !on);
  $("filters").querySelectorAll("input").forEach(i => i.disabled = !on);
}
$("use-filters").onchange = syncFilters;
syncFilters();

$("go2").onclick = () => run($("r2"), $("go2"), async () => {
  const notas = {};
  document.querySelectorAll("#form2 input").forEach(i => {
    if (i.value !== "") notas[i.dataset.id] = Number(i.value);
  });
  const filtros = {};
  if ($("use-filters").checked) [["jogadores", "f-players"], ["minutos", "f-minutes"], ["idade", "f-age"],
   ["complexidade_min", "f-wmin"], ["complexidade_max", "f-wmax"]].forEach(([key, id]) => {
    if ($(id).value !== "") filtros[key] = Number($(id).value);
  });
  showList($("r2"), await api("/api/novo", {notas, filtros}));
});

setupSearch($("s2"), $("h2"), g => addRating(g));
// Campos numericos do item novo e a chave que a API espera para cada um.
const ITEM_FIELDS = [["averageweight", "i-weight"], ["playingtime", "i-minutes"],
  ["minplayers", "i-pmin"], ["maxplayers", "i-pmax"], ["minage", "i-age"]];

setupSearch($("s3"), $("h3"), g => {
  ignoreGame = g.id;
  $("name3").value = g.nome;
  [["cat3", g.categorias], ["mec3", g.mecanicas]].forEach(([id, chosen]) => {
    [...$(id).options].forEach(o => o.selected = chosen.includes(o.value));
  });
  ITEM_FIELDS.forEach(([key, id]) => $(id).value = g[key] ?? "");
  drawRated(ratedGames);
});

$("go3").onclick = () => run($("r3"), $("go3"), async () => {
  const selected = id => [...$(id).selectedOptions].map(o => o.value);
  const nome = $("name3").value.trim();
  const d = await api("/api/item", {
    usuario: $("u3").value.trim(),
    categorias: selected("cat3"),
    mecanicas: selected("mec3"),
    numeros: Object.fromEntries(ITEM_FIELDS.map(([key, id]) => [key, $(id).value === "" ? null : Number($(id).value)])),
    ignorar: ignoreGame,
  });
  const parts = [el("p", d.resumo)];
  if (d.parecidos) parts.push(el("h3", "Jogos do usuário usados na previsão"), makeTable(
    ["Jogo", "Nota do usuário", "Similaridade"],
    d.parecidos.map(x => [x.nome, x.nota.toLocaleString("pt-BR", {minimumFractionDigits: 1}), fmt(x.similaridade)])));
  parts.unshift(el("h3", nome));
  $("r3").replaceChildren(...parts);
});

// Mudar as caracteristicas a mao deixa de simular o jogo escolhido.
["cat3", "mec3", ...ITEM_FIELDS.map(([, id]) => id)]
  .forEach(id => $(id).addEventListener("change", () => { ignoreGame = null; drawRated(ratedGames); }));

api("/api/opcoes").then(o => {
  o.populares.forEach(g => addRating(g));
  [["cat3", o.categorias], ["mec3", o.mecanicas]].forEach(([id, items]) => {
    items.forEach(v => $(id).appendChild(new Option(v, v)));
  });
});
</script>
</body>
</html>
"""


def br(value, decimals=0):
    """Numero no formato brasileiro: 1.234 e 6,88."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


class BadRequest(Exception):
    pass


class Recommender:
    """Dados carregados uma vez e as respostas de cada rota da API."""

    def __init__(self):
        print("Carregando os dados...")
        self.data = rec.load_ratings()
        self.games = rec.load_game_info()
        self.lower_names = self.games["name"].str.lower()
        n_ratings = np.diff(self.data.by_user.indptr)
        self.active_users = np.flatnonzero(n_ratings >= MIN_RATINGS_RANDOM_USER)
        self.by_reviews = self.data.game_ids[np.argsort(-np.diff(self.data.by_game.indptr), kind="stable")]
        # Nomes em minusculas e ordenados, para achar por prefixo com busca binaria.
        self.n_ratings = n_ratings
        lower = np.char.lower(self.data.users)
        self.user_order = np.argsort(lower, kind="stable")
        self.sorted_lower = lower[self.user_order]

    def game_json(self, game_id):
        info = self.games.loc[game_id]
        year = info["yearpublished"]
        return {
            "id": int(game_id),
            "nome": info["name"],
            "ano": None if math.isnan(year) else int(year),
            "categorias": info["boardgamecategory"],
            "mecanicas": info["boardgamemechanic"],
            **{col: None if math.isnan(info[col]) else round(float(info[col]), 2) for col in rec.NUMERIC_FEATURES},
        }

    def options(self):
        unique = lambda col: sorted({x for values in self.games[col] for x in values})
        return {
            "categorias": unique("boardgamecategory"),
            "mecanicas": unique("boardgamemechanic"),
            "populares": [self.game_json(g) for g in self.by_reviews[:POPULAR_FORM_SIZE].tolist()],
        }

    def search(self, query):
        found = self.games[self.lower_names.str.contains(query.lower(), regex=False)]
        found = found.sort_values("board_game_rank").head(SEARCH_LIMIT)
        return [self.game_json(g) for g in found.index]

    def search_users(self, query):
        """Usuarios cujo nome comeca com query (sem diferenciar maiusculas), os mais ativos primeiro."""
        query = query.strip().lower()
        if not query:
            return []
        start, end = np.searchsorted(self.sorted_lower, [query, query + "\U0010ffff"])
        matches = self.user_order[start:end]
        if len(matches) > USER_SEARCH_LIMIT:
            matches = matches[np.argpartition(-self.n_ratings[matches], USER_SEARCH_LIMIT)[:USER_SEARCH_LIMIT]]
        matches = sorted(matches.tolist(), key=lambda u: (-self.n_ratings[u], self.data.users[u]))
        return [{"usuario": str(self.data.users[u]), "avaliacoes": int(self.n_ratings[u])} for u in matches]

    def random_user(self):
        return {"usuario": str(self.data.users[random.choice(self.active_users.tolist())])}

    def user_ratings(self, name):
        """Jogos que o usuario avaliou, da maior para a menor nota."""
        user = self.require_user(name)
        row = self.data.by_user[user]
        games = [
            {"id": int(g), "nome": self.games.at[g, "name"], "nota": float(r)}
            for g, r in zip(self.data.game_ids[row.indices].tolist(), row.data.tolist())
        ]
        games.sort(key=lambda g: (-g["nota"], g["nome"]))
        return {"usuario": str(self.data.users[user]), "media": float(self.data.means[user]), "jogos": games}

    def require_user(self, name):
        user = rec.find_user(self.data, name or "")
        if user is None:
            similar = rec.suggest_users(self.data, name or "")
            hint = f" Parecidos: {', '.join(similar)}." if similar else ""
            raise BadRequest(f"Usuário '{name}' não encontrado.{hint}")
        return user

    def list_json(self, table, summary, profile=None):
        items = []
        for row in table.itertuples():
            game = self.game_json(row.game_id)
            items.append({
                "nome": game["nome"],
                "ano": game["ano"],
                "nota": None if math.isnan(row.nota) else float(row.nota),
                "similaridade": None if math.isnan(row.similaridade) else float(row.similaridade),
                "pontuacao": None if math.isnan(row.pontuacao) else float(row.pontuacao),
                "metodo": row.metodo,
            })
        return {"resumo": summary, "perfil": profile, "itens": items}

    def old_user(self, body):
        user = self.require_user(body.get("usuario"))
        table, _, n_similar = rec.recommend_existing_user(self.data, self.games, user)
        profile = {
            "avaliacoes": int(self.data.by_user[user].nnz),
            "media": float(self.data.means[user]),
            "desvio": float(self.data.stds[user]),
            "similares": n_similar,
        }
        summary = None
        if self.data.stds[user] < rec.EPS:
            summary = "Este usuário deu a mesma nota a tudo: sem Pearson possível, a lista vem da popularidade."
        return self.list_json(table, summary, profile)

    def new_user(self, body):
        ratings = {}
        for game_id, value in body.get("notas", {}).items():
            try:
                game_id, value = int(game_id), float(value)
            except (TypeError, ValueError):
                raise BadRequest("Notas inválidas.")
            if game_id not in self.games.index:
                raise BadRequest(f"Jogo {game_id} não existe.")
            if not rec.RATING_MIN <= value <= rec.RATING_MAX:
                raise BadRequest("As notas precisam estar entre 1 e 10.")
            ratings[game_id] = value
        filters = self.parse_filters(body.get("filtros", {}))
        n_allowed = None
        if filters:
            n_allowed = int(rec.knowledge_filter(self.data, self.games, **filters).sum())
            if n_allowed == 0:
                raise BadRequest("Nenhum jogo atende a esses filtros.")
        table, n_scored, n_similar = rec.recommend_new_user(self.data, self.games, ratings, filters=filters)
        if n_similar:
            summary = f"{len(ratings)} notas. {br(n_similar)} usuários similares; {br(n_scored)} jogos com nota."
        else:
            summary = (
                f"{len(ratings)} notas: sem vizinhos para calcular (são necessárias ao menos "
                f"{rec.MIN_COMMON} notas diferentes entre si). Lista por popularidade."
            )
        if n_allowed is not None:
            summary += f" {br(n_allowed)} jogos atendem aos filtros."
        return self.list_json(table, summary)

    def parse_filters(self, raw):
        """Converte os filtros da tela nos argumentos de rec.knowledge_filter."""
        try:
            values = {k: float(v) for k, v in raw.items() if v is not None}
        except (AttributeError, TypeError, ValueError):
            raise BadRequest("Filtros inválidos.")
        filters = {}
        if "jogadores" in values:
            filters["players"] = values["jogadores"]
        if "minutos" in values:
            filters["max_minutes"] = values["minutos"]
        if "idade" in values:
            filters["age"] = values["idade"]
        if "complexidade_min" in values or "complexidade_max" in values:
            low, high = values.get("complexidade_min", 1.0), values.get("complexidade_max", 5.0)
            if low > high:
                raise BadRequest("A complexidade mínima é maior que a máxima.")
            filters["weight_range"] = (low, high)
        return filters

    def new_item(self, body):
        user = self.require_user(body.get("usuario"))
        categories, mechanics = body.get("categorias", []), body.get("mecanicas", [])
        if not categories and not mechanics:
            raise BadRequest("Escolha ao menos uma categoria ou mecânica.")
        try:
            numbers = {col: None if v is None else float(v)
                       for col, v in body.get("numeros", {}).items() if col in rec.NUMERIC_FEATURES}
        except (AttributeError, TypeError, ValueError):
            raise BadRequest("Características numéricas inválidas.")
        if None not in (numbers.get("minplayers"), numbers.get("maxplayers")) \
                and numbers["minplayers"] > numbers["maxplayers"]:
            raise BadRequest("O mínimo de jogadores é maior que o máximo.")
        ignore = body.get("ignorar")
        item = {"categories": categories, "mechanics": mechanics, **numbers}
        result = rec.predict_new_item(self.data, self.games, user, item, ignore)
        if result is None:
            return {"resumo": "Nenhum jogo do usuário tem categoria ou mecânica em comum: não há como prever."}
        verdict = "Recomendar" if result["recomendar"] else "Não recomendar"
        summary = (
            f"{verdict}: nota prevista {br(result['nota'], 2)}, "
            f"média do usuário {br(result['media_usuario'], 2)}."
        )
        if ignore is not None:
            real = self.data.by_user[user, np.searchsorted(self.data.game_ids, ignore)]
            if real:
                summary += f" Nota real do usuário neste jogo: {br(float(real), 1)}."
        similar = [
            {"nome": self.games.loc[p["game_id"], "name"], "nota": p["nota"], "similaridade": p["similaridade"]}
            for p in result["parecidos"]
        ]
        return {"resumo": summary, "parecidos": similar}


def make_handler(recommender):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def respond(self, action):
            try:
                self.send(200, json.dumps(action(), ensure_ascii=False))
            except BadRequest as err:
                self.send(400, json.dumps({"erro": str(err)}, ensure_ascii=False))
            except Exception as err:
                self.log_error("%r", err)
                self.send(500, json.dumps({"erro": "Erro interno; veja o terminal."}, ensure_ascii=False))

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.send(200, PAGE, "text/html; charset=utf-8")
            elif url.path == "/api/opcoes":
                self.respond(recommender.options)
            elif url.path == "/api/random-user":
                self.respond(recommender.random_user)
            elif url.path == "/api/usuarios":
                query = parse_qs(url.query).get("q", [""])[0]
                self.respond(lambda: recommender.search_users(query))
            elif url.path == "/api/avaliacoes":
                name = parse_qs(url.query).get("usuario", [""])[0]
                self.respond(lambda: recommender.user_ratings(name))
            elif url.path == "/api/search":
                query = parse_qs(url.query).get("q", [""])[0]
                self.respond(lambda: recommender.search(query))
            else:
                self.send(404, json.dumps({"erro": "Não encontrado."}, ensure_ascii=False))

        def do_POST(self):
            routes = {
                "/api/antigo": recommender.old_user,
                "/api/novo": recommender.new_user,
                "/api/item": recommender.new_item,
            }
            route = routes.get(self.path)
            if route is None:
                self.send(404, json.dumps({"erro": "Não encontrado."}, ensure_ascii=False))
                return

            def action():
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    raise BadRequest("JSON inválido.")
                return route(body)

            self.respond(action)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Interface web do recomendador.")
    parser.add_argument("--porta", type=int, default=8000, help="porta do servidor (padrao: 8000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.porta), make_handler(Recommender()))
    print(f"Pronto: http://127.0.0.1:{args.porta}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
