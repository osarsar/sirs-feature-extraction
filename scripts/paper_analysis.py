"""
Reproduce every human-evaluation number reported in the paper from the released files.

Inputs (all in this repository):
  data/human_ratings_16.csv            anonymised ratings of the 16 annotators (0.15 / 0.50 / 0.85)
  data/gold_test_with_dimensions.json  26 test claims + LLM dimension scores
  results/predictions_v4.json          CamemBERT test/val predictions, 3 seeds (train_corrected_v4.py)
  results/weights_val_optimised.json   weights re-fitted on validation (optimize_weights_val.py)

Outputs (printed):
  - inter-annotator agreement (Krippendorff alpha, pairwise rho, leave-one-out reference level)
  - Table 5  (main results, published SIRS weights) with 95% bootstrap CIs
  - Table 6  (paired bootstrap differences)
  - Table 8  (re-fitted weights)
  - Section 6.5 (median instead of mean)

Usage:  python3 scripts/paper_analysis.py
Requires: numpy, scipy, krippendorff (pip install krippendorff)
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sirs_paper1 import FeatureExtractor, PAPER1_WEIGHTS  # noqa: E402

DIMS = ['x1', 'x2', 'x3', 'x4']
B = 10_000
SEED = 42


def rho(a, b):
    r = spearmanr(a, b)[0]
    return 0.0 if np.isnan(r) else float(r)


def score(dims, w):
    return np.minimum(1.0, np.array([dims[d] for d in DIMS]).T @ np.array(w))


def load():
    with open(ROOT / "data" / "human_ratings_16.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    A = np.array([[float(v) for v in r[1:]] for r in rows[1:]]).T  # (annotators, claims)
    gold = json.load(open(ROOT / "data" / "gold_test_with_dimensions.json"))
    preds = json.load(open(ROOT / "results" / "predictions_v4.json"))
    W = json.load(open(ROOT / "results" / "weights_val_optimised.json"))
    return A, gold, preds, W


def gold_mean(X):
    # Round so that claims with identical vote counts are exact ties (avoids floating-point
    # noise such as 0.5375000000000001 vs 0.5374999999999999 breaking ties in Spearman).
    return np.round(np.mean(X, 0), 9)


def agreement(A):
    import krippendorff
    k = A.shape[0]
    alpha = krippendorff.alpha(reliability_data=A, level_of_measurement='interval')
    pw = [rho(A[i], A[j]) for i, j in itertools.combinations(range(k), 2)]
    loo = [rho(A[i], gold_mean(np.delete(A, i, 0))) for i in range(k)]
    print(f"Annotators: {k}, claims: {A.shape[1]}")
    print(f"Krippendorff alpha (interval): {alpha:.3f}")
    print(f"Mean pairwise Spearman: {np.mean(pw):.3f}  (range {min(pw):.2f} to {max(pw):.2f})")
    print(f"Leave-one-out reference level (vs mean of others): {np.mean(loo):.3f}")


def methods(gold, preds, W):
    llm = {d: [g[f'llm_{d}'] for g in gold] for d in DIMS}
    fe = FeatureExtractor()
    kw = [fe.extract_numeric(g['claim']) for g in gold]
    kwd = {d: [k[i] for k in kw] for i, d in enumerate(DIMS)}
    P1 = tuple(PAPER1_WEIGHTS)
    M = {
        'Rule-based (Paper 1)': [score(kwd, P1)],
        'CamemBERT separate': [score(r['separate']['test'], P1) for r in preds],
        'CamemBERT unified': [score(r['unified']['test'], P1) for r in preds],
        'LLM zero-shot': [score(llm, P1)],
        'CamemBERT separate (re-fitted w)': [score(r['separate']['test'], tuple(W['per_seed'][i]['separate']['weights_val_opt'])) for i, r in enumerate(preds)],
        'CamemBERT unified (re-fitted w)': [score(r['unified']['test'], tuple(W['per_seed'][i]['unified']['weights_val_opt'])) for i, r in enumerate(preds)],
        'LLM zero-shot (re-fitted w)': [score(llm, tuple(e['llm']['weights_val_opt'])) for e in W['per_seed']],
    }
    return M


def evaluate(y, M, label):
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    boots = {}
    print(f"\n=== {label} ===")
    print(f"{'Method':<34} {'MSE':>6} {'MAE':>6} {'rho':>14} {'95% CI':>16}")
    for k, ps in M.items():
        rs = [rho(y, p) for p in ps]
        boots[k] = np.array([np.mean([rho(y[i], p[i]) for p in ps]) for i in idx])
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        mse = np.mean([np.mean((p - y) ** 2) for p in ps])
        mae = np.mean([np.mean(np.abs(p - y)) for p in ps])
        sd = f"±{np.std(rs):.3f}" if len(rs) > 1 else ""
        print(f"{k:<34} {mse:6.3f} {mae:6.3f} {np.mean(rs):8.3f}{sd:>6} [{lo:5.2f}, {hi:5.2f}]")
    pairs = [('LLM zero-shot', 'Rule-based (Paper 1)'), ('CamemBERT separate', 'Rule-based (Paper 1)'),
             ('CamemBERT unified', 'Rule-based (Paper 1)'), ('LLM zero-shot', 'CamemBERT separate'),
             ('CamemBERT separate', 'CamemBERT unified'),
             ('CamemBERT separate (re-fitted w)', 'CamemBERT separate'),
             ('CamemBERT unified (re-fitted w)', 'CamemBERT unified'),
             ('LLM zero-shot (re-fitted w)', 'LLM zero-shot'),
             ('LLM zero-shot (re-fitted w)', 'CamemBERT separate (re-fitted w)'),
             ('CamemBERT separate (re-fitted w)', 'CamemBERT unified (re-fitted w)')]
    print("\nPaired bootstrap differences:")
    for a, b in pairs:
        d = boots[a] - boots[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {a} − {b}: Δ={d.mean():+.3f} [{lo:+.2f}, {hi:+.2f}]  P(Δ≤0)={np.mean(d <= 0):.3f}")


def main():
    A, gold, preds, W = load()
    agreement(A)
    M = methods(gold, preds, W)
    evaluate(gold_mean(A), M, "Gold = mean of 16 annotators (main analysis)")
    evaluate(np.median(A, 0), M, "Gold = median of 16 annotators (robustness, Section 6.5)")


if __name__ == "__main__":
    main()
