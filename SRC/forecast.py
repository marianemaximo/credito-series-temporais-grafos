"""Previsão walk-forward da variação mensal da inadimplência, com e sem features de grafos de quantis.

Alvo: Δy_{t+1} = y_{t+1} - y_t (variação no mês seguinte). Em cada mês t (a partir de `t_start`) os modelos são
reestimados SÓ com linhas cujo alvo já era conhecido em t, e preveem Δy_{t+1}. Todas as features de um mês t usam
apenas y[..t] (inclusive o grafo de quantis, construído na janela y[t-window+1..t]).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .qgraph import qg_features, quantile_graph

QG_COLS = ["self_loop", "mjl", "entropy", "modularity", "path_len", "node_cur", "p_up", "p_down", "exp_shift", "qg_fc"]
BASE_COLS = ["d1", "d2", "d3", "d12", "z"]
SEAS_COLS = ["s1", "c1", "s2", "c2"]


def load_series(path) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",", quotechar='"')
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    return df.set_index("data")["valor"].astype(float).sort_index()


def make_features(y: pd.Series, window: int = 48, Q: int = 4, h: int = 1, qg_on: str = "level") -> pd.DataFrame:
    """Tabela de features indexada pelo índice posicional t do último mês observado.

    Alvo: y[t+h] - y[t]. As h últimas linhas não têm alvo (NaN).
    """
    v, idx = y.values, y.index
    n = len(v)
    rows = []
    start = max(window - 1, 12) if qg_on == "level" else max(window, 12)
    for t in range(start, n):
        w = v[t - window + 1: t + 1]
        if qg_on == "level":
            f = qg_features(w, Q)
        else:                                   # QG sobre as variações mensais da janela (sensibilidade)
            wd = np.diff(v[t - window: t + 1])
            f = qg_features(wd, Q)
            g = quantile_graph(wd, Q)
            nm = np.where(np.isnan(g.node_mean), 0.0, g.node_mean)
            f["qg_fc"] = f["qg_fc_node"] = float((g.P[int(g.labels[-1])] * nm).sum())   # Δ esperado do próximo mês
        tm = (idx[t].month - 1 + h) % 12 + 1              # mês do alvo (t+h)
        ang = 2 * np.pi * tm / 12
        f.update({
            "t": t, "date": idx[t],
            "d1": v[t] - v[t - 1], "d2": v[t - 1] - v[t - 2], "d3": v[t - 2] - v[t - 3], "d12": v[t] - v[t - 12],
            "z": (v[t] - w.mean()) / (w.std() + 1e-9),
            "s1": np.sin(ang), "c1": np.cos(ang), "s2": np.sin(2 * ang), "c2": np.cos(2 * ang),
            "tmonth": tm, "y": v[t],
            "target": v[t + h] - v[t] if t + h < n else np.nan,
        })
        rows.append(f)
    return pd.DataFrame(rows).set_index("t")


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 15)))


def _rf(seed=0):
    return RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=5, n_jobs=2, random_state=seed)


MODELS = {
    # nome: (tipo, colunas)
    "Passeio aleatório (Δ=0)": ("zero", []),
    "Drift": ("drift", []),
    "Sazonal (média do mês)": ("season", []),
    "AR + sazonal (Ridge)": ("ridge", BASE_COLS + SEAS_COLS),
    "QG-Markov (direto)": ("direct", ["qg_fc"]),
    "QG-Markov nó→nó (pós-hoc)": ("direct", ["qg_fc_node"]),
    "QG apenas (Ridge)": ("ridge", QG_COLS),
    "AR + sazonal + QG (Ridge)": ("ridge", BASE_COLS + SEAS_COLS + QG_COLS),
    "AR + sazonal (RF)": ("rf", BASE_COLS + SEAS_COLS),
    "AR + sazonal + QG (RF)": ("rf", BASE_COLS + SEAS_COLS + QG_COLS),
}


def _fit_predict(kind, cols, train, test_row):
    if kind == "zero":
        return 0.0
    if kind == "drift":
        return float(train["target"].mean())
    if kind == "season":
        m = train.groupby("tmonth")["target"].mean()
        return float(m.get(int(test_row["tmonth"]), train["target"].mean()))
    if kind == "direct":
        return float(test_row[cols[0]])
    model = _ridge() if kind == "ridge" else _rf()
    model.fit(train[cols].values, train["target"].values)
    return float(model.predict(test_row[cols].values.reshape(1, -1))[0])


def walk_forward(feat: pd.DataFrame, t_start: int, models: dict | None = None, min_train: int = 36, h: int = 1) -> pd.DataFrame:
    """Previsões fora da amostra, mês a mês. Retorna DataFrame (index = t) com 'actual' e uma coluna por modelo."""
    models = models or MODELS
    labeled = feat[feat["target"].notna()]
    out = []
    for t in labeled.index[labeled.index >= t_start]:
        train = labeled[labeled.index <= t - h]          # alvo da linha s só é conhecido em s+h <= t
        if len(train) < min_train:
            continue
        row = labeled.loc[t]
        rec = {"t": t, "date": row["date"], "actual": row["target"], "y": row["y"]}
        for name, (kind, cols) in models.items():
            rec[name] = _fit_predict(kind, cols, train, row)
        out.append(rec)
    return pd.DataFrame(out).set_index("t")


def arima_walk_forward(y: pd.Series, t_start: int, t_end: int, order=(1, 1, 1), h: int = 1) -> pd.Series:
    from statsmodels.tsa.arima.model import ARIMA
    out = {}
    for t in range(t_start, t_end + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y.values[: t + 1], order=order).fit()
            out[t] = float(res.forecast(h)[-1] - y.values[t])
    return pd.Series(out, name="ARIMA(1,1,1)")


# ---------------------------------------------------------------------- métricas e testes
def holm(pvals):
    """Correção de Holm-Bonferroni (controle do erro por família). Retorna p-valores ajustados na ordem original."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def direction_accuracy(actual, pred):
    a, p = np.sign(actual), np.sign(pred)
    m = a != 0
    return float((a[m] == p[m]).mean())


def score_table(res: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    a = res["actual"].values
    cols = [c for c in res.columns if c not in ("actual", "date", "y")]
    preds = {c: res[c].values for c in cols}
    if extra:
        preds.update({k: v for k, v in extra.items()})
    rw = rmse(a)
    rows = []
    for k, p in preds.items():
        e = a - p
        rows.append({"modelo": k, "RMSE": rmse(e), "MAE": float(np.mean(np.abs(e))), "RMSE_rel_RW": rmse(e) / rw,
                     "acerto_direcao": np.nan if np.allclose(p, 0) else direction_accuracy(a, p)})
    return pd.DataFrame(rows).set_index("modelo")


def diebold_mariano(e1, e2, h: int = 1):
    """Teste DM (perda quadrática) com correção de Harvey-Leybourne-Newbold. H0: previsões igualmente precisas.

    Retorna (estatística, p-valor bicaudal); estatística < 0 => modelo 1 tem erro menor.
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    g0 = np.var(d, ddof=0)
    lrv = g0
    for k in range(1, h):
        lrv += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    lrv = max(lrv, 1e-12)                       # HLN não-ponderado pode ficar <= 0 em amostras pequenas
    dm = dbar / np.sqrt(lrv / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


def block_bootstrap_ratio(e_a, e_b, block: int = 6, n_boot: int = 2000, seed: int = 0):
    """IC 95% (bootstrap em blocos móveis) para RMSE_A / RMSE_B; < 1 => A melhor."""
    rng = np.random.default_rng(seed)
    e_a, e_b = np.asarray(e_a), np.asarray(e_b)
    n = len(e_a)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out.append(rmse(e_a[idx]) / rmse(e_b[idx]))
    return rmse(e_a) / rmse(e_b), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
