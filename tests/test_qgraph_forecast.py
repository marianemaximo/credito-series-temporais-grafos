import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecast import (MODELS, block_bootstrap_ratio, diebold_mariano, holm, load_series, make_features,
                          walk_forward)
from src.qgraph import (fbm, hurst_qg, mean_jump_length, phase_surrogate, quantile_graph, self_loop_mass,
                        transition_entropy)

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "bcdata.sgs.21082.csv"


def test_qg_recovers_known_symmetric_markov_chain():
    rng = np.random.default_rng(0)
    Pt = np.array([[.7, .3, 0, 0], [.3, .4, .3, 0], [0, .3, .4, .3], [0, 0, .3, .7]])  # ocupação uniforme
    s = [0]
    for _ in range(40000):
        s.append(rng.choice(4, p=Pt[s[-1]]))
    x = np.array(s) + rng.uniform(-.1, .1, len(s))
    g = quantile_graph(x, 4)
    assert np.abs(g.P - Pt).max() < 0.03
    assert np.allclose(g.P.sum(axis=1), 1)


def test_qg_metrics_separate_periodic_from_random():
    rng = np.random.default_rng(1)
    per = np.sin(2 * np.pi * np.arange(400) / 20)
    ale = rng.standard_normal(400)
    gp, ga = quantile_graph(per, 5), quantile_graph(ale, 5)
    assert self_loop_mass(gp) > self_loop_mass(ga) + 0.2
    assert mean_jump_length(gp) < mean_jump_length(ga)
    assert transition_entropy(gp) < transition_entropy(ga)


def test_hurst_qg_orders_h_values_on_fbm():
    rng = np.random.default_rng(2)
    est = {H: np.mean([hurst_qg(fbm(187, H, rng), Q=10)[0] for _ in range(25)]) for H in (0.3, 0.5, 0.8)}
    assert est[0.3] < est[0.5] < est[0.8]
    assert abs(est[0.3] - 0.3) < 0.1


def test_phase_surrogate_preserves_spectrum():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(256))
    s = phase_surrogate(x, rng)
    assert np.allclose(np.abs(np.fft.rfft(x - x.mean())), np.abs(np.fft.rfft(s - s.mean())), rtol=1e-6, atol=1e-6)


def test_no_lookahead_in_features_and_predictions():
    """Alterar valores FUTUROS não pode mudar features nem previsões já feitas."""
    y = load_series(DATA)
    y2 = y.copy()
    cut = 150
    y2.iloc[cut + 1:] += np.random.default_rng(0).normal(0, 1.0, len(y2) - cut - 1)   # perturba o futuro
    F1, F2 = make_features(y, 48, 4, h=1), make_features(y2, 48, 4, h=1)
    feat_cols = [c for c in F1.columns if c not in ("target", "date")]
    assert np.allclose(F1.loc[:cut, feat_cols].values, F2.loc[:cut, feat_cols].values)
    small = {k: MODELS[k] for k in ("Sazonal (média do mês)", "AR + sazonal + QG (Ridge)")}
    R1 = walk_forward(F1, 95, models=small)
    R2 = walk_forward(F2, 95, models=small)
    common = R1.index[R1.index <= cut - 1]          # previsões cujo alvo (t+1) ainda é <= cut
    assert len(common) > 20
    for k in small:
        assert np.allclose(R1.loc[common, k], R2.loc[common, k])


def test_horizon_training_excludes_unrealized_targets():
    y = load_series(DATA)
    h = 6
    F = make_features(y, 48, 4, h=h)
    lab = F[F["target"].notna()]
    assert lab.index.max() == len(y) - 1 - h


def test_dm_and_holm_and_bootstrap_sanity():
    rng = np.random.default_rng(4)
    e_good, e_bad = rng.normal(0, 1, 200), rng.normal(0, 2, 200)
    dm, p = diebold_mariano(e_good, e_bad)
    assert dm < 0 and p < 0.001
    dm, p = diebold_mariano(e_good, rng.normal(0, 1, 200))
    assert p > 0.05
    r, lo, hi = block_bootstrap_ratio(e_good, e_bad)
    assert hi < 1
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
