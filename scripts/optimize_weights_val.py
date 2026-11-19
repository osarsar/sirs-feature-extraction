"""
Re-optimise the SIRS formula weights on the VALIDATION split (LLM silver risk
labels), then apply them to the 26-claim human gold test set.

Same grid as Paper 1 (Sec. 5.2): w_i in {0.05, 0.10, ..., 0.60}, sum(w) = 1.
The human test set is NEVER used to choose the weights.

Input : results/predictions_v4.json  (from train_corrected_v4.py)
Output: results/weights_val_optimised.json

Usage : python3 scripts/optimize_weights_val.py [path/to/predictions_v4.json]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
PRED_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "predictions_v4.json"
GOLD_FILE = ROOT / "data" / "gold_test_with_dimensions.json"
OUT_FILE = ROOT / "results" / "weights_val_optimised.json"

PAPER1_WEIGHTS = (0.20, 0.10, 0.55, 0.15)
DIMS = ['x1', 'x2', 'x3', 'x4']


def grid():
    steps = [round(0.05 * k, 2) for k in range(1, 13)]  # 0.05 .. 0.60
    return [w for w in itertools.product(steps, repeat=4) if abs(sum(w) - 1.0) < 1e-9]


def score(dims, w):
    X = np.array([dims[d] for d in DIMS]).T  # (n, 4)
    return np.minimum(1.0, X @ np.array(w))


def rho(y, p):
    r = spearmanr(y, p)[0]
    return 0.0 if np.isnan(r) else float(r)


def best_on_val(val_dims, val_target, W):
    vals = [rho(val_target, score(val_dims, w)) for w in W]
    i = int(np.argmax(vals))
    return W[i], vals[i]


def main():
    W = grid()
    print(f"Grid: {len(W)} weight combinations (Paper 1 grid)")
    runs = json.load(open(PRED_FILE))
    gold = json.load(open(GOLD_FILE))
    y_test = [g['risk_score'] for g in gold]
    llm_test = {d: [g[f'llm_{d}'] for g in gold] for d in DIMS}

    results = {'grid_size': len(W), 'paper1_weights': PAPER1_WEIGHTS, 'per_seed': []}
    summary = {k: {'paper1': [], 'val_opt': []} for k in ['separate', 'unified', 'llm']}

    for run in runs:
        y_val = run['val_llm_risk']
        entry = {'seed': run['seed']}
        candidates = {
            'separate': (run['separate']['val'], run['separate']['test']),
            'unified': (run['unified']['val'], run['unified']['test']),
            # LLM: its own dimension scores on val (silver) → same procedure
            'llm': (run['val_llm_dims'], llm_test),
        }
        for name, (val_dims, test_dims) in candidates.items():
            w, val_rho = best_on_val(val_dims, y_val, W)
            r_p1 = rho(y_test, score(test_dims, PAPER1_WEIGHTS))
            r_opt = rho(y_test, score(test_dims, w))
            entry[name] = {'weights_val_opt': w, 'val_rho': val_rho,
                           'test_rho_paper1_w': r_p1, 'test_rho_val_opt_w': r_opt}
            summary[name]['paper1'].append(r_p1)
            summary[name]['val_opt'].append(r_opt)
        results['per_seed'].append(entry)

    print(f"\n{'Method':<10} {'Paper 1 weights':>18} {'Val-optimised weights':>24}   weights chosen per seed")
    for name in ['separate', 'unified', 'llm']:
        a, b = np.array(summary[name]['paper1']), np.array(summary[name]['val_opt'])
        ws = [e[name]['weights_val_opt'] for e in results['per_seed']]
        print(f"{name:<10} {a.mean():>10.3f} ± {a.std():.3f} {b.mean():>16.3f} ± {b.std():.3f}   {ws}")
        results[name] = {'paper1_mean': float(a.mean()), 'paper1_std': float(a.std()),
                         'val_opt_mean': float(b.mean()), 'val_opt_std': float(b.std())}

    OUT_FILE.parent.mkdir(exist_ok=True)
    json.dump(results, open(OUT_FILE, 'w'), indent=2)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
