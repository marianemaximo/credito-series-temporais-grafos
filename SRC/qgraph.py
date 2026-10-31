"""Grafos de quantis (Quantile Graphs, QG).

Mapa série temporal -> rede proposto por:
    Campanharo, Sirer, Malmgren, Ramos & Amaral (2011). "Duality between Time Series and Networks".
    PLoS ONE 6(8): e23378.
e usado em Campanharo & Ramos (2016), Physica A 444:43-48 (expoente de Hurst), e Campanharo et al. (2018).

Construção: a série é dividida em Q quantis; cada quantil é um nó; o peso do arco i -> j é a
probabilidade de um ponto no quantil i ser seguido por um ponto no quantil j (matriz de transição P).
A rede é ponderada, dirigida e tem laços (self-loops).

Observação: as funções de métricas de rede e a previsão markoviana abaixo são implementações próprias
sobre essa construção; a estimativa de Hurst segue a ideia descrita no resumo de Campanharo & Ramos (2016)
(expoente de escala do comprimento médio de salto para diferentes defasagens) e é validada em fBm simulado
nos testes — não é uma reprodução linha a linha do artigo.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class QG:
    labels: np.ndarray      # nó (0..Q-1) de cada observação
    W: np.ndarray           # contagem de transições (Q x Q)
    P: np.ndarray           # probabilidades de transição por linha
    occ: np.ndarray         # ocupação de cada nó (fração dos pontos de origem)
    node_mean: np.ndarray   # valor médio da série em cada nó
    edges: np.ndarray       # limites internos dos quantis
    Q: int


def assign_nodes(x, edges):
    return np.searchsorted(edges, np.asarray(x), side="right")


def quantile_graph(x, Q: int = 4, lag: int = 1, edges: np.ndarray | None = None) -> QG:
    """Constrói o QG de `x`. `lag` > 1 liga o ponto t ao ponto t+lag (usado para estimar Hurst)."""
    x = np.asarray(x, dtype=float)
    if edges is None:
        edges = np.quantile(x, np.linspace(0, 1, Q + 1)[1:-1])
    lab = assign_nodes(x, edges)
    W = np.zeros((Q, Q))
    np.add.at(W, (lab[:-lag], lab[lag:]), 1.0)
    rows = W.sum(axis=1, keepdims=True)
    P = np.where(rows > 0, W / np.where(rows > 0, rows, 1), 0.0)
    for i in np.where(rows[:, 0] == 0)[0]:       # nó sem saída (só o último ponto): laço
        P[i, i] = 1.0
    occ = rows[:, 0] / max(rows.sum(), 1.0)
    node_mean = np.array([x[lab == i].mean() if (lab == i).any() else np.nan for i in range(Q)])
    return QG(lab, W, P, occ, node_mean, edges, Q)


# ----------------------------------------------------------------------------- métricas de rede
def mean_jump_length(g: QG) -> float:
    """Comprimento médio de salto (em nós): E[|j - i|] sob a ocupação e as transições do grafo."""
    i, j = np.indices(g.P.shape)
    return float((g.occ[:, None] * g.P * np.abs(i - j)).sum())


def self_loop_mass(g: QG) -> float:
    """Persistência: probabilidade de permanecer no mesmo quantil (traço ponderado de P)."""
    return float((g.occ * np.diag(g.P)).sum())


def transition_entropy(g: QG) -> float:
    """Entropia média (normalizada em [0,1]) das linhas de P: quão previsível é o próximo quantil."""
    p = np.clip(g.P, 1e-12, 1)
    h = -(g.P * np.log(p)).sum(axis=1) / np.log(g.Q)
    return float((g.occ * h).sum())


def _undirected(g: QG) -> nx.Graph:
    A = g.W + g.W.T
    G = nx.Graph()
    G.add_nodes_from(range(g.Q))
    for i in range(g.Q):
        for j in range(i + 1, g.Q):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=float(A[i, j]), dist=1.0 / float(A[i, j]))
    return G


def modularity(g: QG) -> float:
    G = _undirected(g)
    if G.number_of_edges() == 0:
        return 0.0
    com = nx.community.greedy_modularity_communities(G, weight="weight")
    return float(nx.community.modularity(G, com, weight="weight"))


def avg_path_length(g: QG) -> float:
    """Comprimento médio de caminho mínimo (distância = 1/peso), apenas entre pares alcançáveis."""
    G = _undirected(g)
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight="dist"))
    vals = [v for u in d for t, v in d[u].items() if t != u]
    return float(np.mean(vals)) if vals else 0.0


def current_node_features(g: QG) -> dict:
    """Condicionais ao nó onde a série está agora (último ponto): o que o grafo diz sobre o próximo passo."""
    c = int(g.labels[-1])
    row = g.P[c]
    j = np.arange(g.Q)
    return {
        "node_cur": c / (g.Q - 1),
        "p_up": float(row[j > c].sum()),
        "p_down": float(row[j < c].sum()),
        "exp_shift": float((row * (j - c)).sum() / (g.Q - 1)),
    }


def qg_markov_forecast_delta(g: QG, x_last: float) -> float:
    """Mapa inverso aproximado: valor esperado do próximo ponto = soma_j P[c,j] * média_do_nó_j; devolve Δ = esperado - atual."""
    c = int(g.labels[-1])
    nm = np.where(np.isnan(g.node_mean), x_last, g.node_mean)
    return float((g.P[c] * nm).sum() - x_last)


def qg_markov_forecast_node_delta(g: QG) -> float:
    """Variante (adicionada POST HOC, após ver que a versão acima erra por ignorar a posição dentro do nó):
    salto esperado entre médias de nós, sum_j P[c,j] * (média_j - média_c)."""
    c = int(g.labels[-1])
    nm = np.where(np.isnan(g.node_mean), g.node_mean[c], g.node_mean)
    return float((g.P[c] * (nm - g.node_mean[c])).sum())


def qg_features(x, Q: int = 4) -> dict:
    """Todas as features de um QG construído sobre a janela `x` (usa apenas dados da janela)."""
    g = quantile_graph(x, Q)
    f = {
        "self_loop": self_loop_mass(g), "mjl": mean_jump_length(g), "entropy": transition_entropy(g),
        "modularity": modularity(g), "path_len": avg_path_length(g),
    }
    f.update(current_node_features(g))
    f["qg_fc"] = qg_markov_forecast_delta(g, float(np.asarray(x)[-1]))
    f["qg_fc_node"] = qg_markov_forecast_node_delta(g)
    return f


# ----------------------------------------------------------------------------- surrogates
def shuffle_surrogate(x, rng):
    return rng.permutation(np.asarray(x))


def phase_surrogate(x, rng):
    """Surrogate por aleatorização de fase: preserva o espectro (autocorrelação linear), destrói não linearidades."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    F = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0
    if n % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n) + x.mean()


# ----------------------------------------------------------------------------- Hurst via QG
def hurst_qg(x, Q: int = 10, lags=range(1, 9)) -> tuple[float, np.ndarray, np.ndarray]:
    """Expoente H como inclinação de log(MJL) vs log(defasagem), com nós definidos na série inteira."""
    x = np.asarray(x, dtype=float)
    edges = np.quantile(x, np.linspace(0, 1, Q + 1)[1:-1])
    lags = np.array(list(lags))
    mjl = np.array([mean_jump_length(quantile_graph(x, Q, lag=int(l), edges=edges)) for l in lags])
    H = np.polyfit(np.log(lags), np.log(mjl), 1)[0]
    return float(H), lags, mjl


def fbm(n: int, H: float, rng) -> np.ndarray:
    """Movimento browniano fracionário por Cholesky (exato; ok para n pequeno)."""
    k = np.arange(n)
    # incrementos fGn com autocovariância exata
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    C = g[np.abs(k[:, None] - k[None, :])]
    inc = np.linalg.cholesky(C + 1e-10 * np.eye(n)) @ rng.standard_normal(n)
    return np.cumsum(inc)
