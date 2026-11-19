"""
Architecture Ablation Study
Tests architecture variants per dimension to select the final design.

Experiments:
  x1 (Topic Sensitivity):
    - Regression only: 1L, 2L, 3L
    - Multi-task (5-class sensitivity level + regression): 1L, 2L, 3L
  x2 (Actionability):
    - Regression only: 1L, 2L, 3L
  x3 (Audience Scope):
    - Regression only: 1L, 2L, 3L
    - Multi-task (3-class scope level + regression): 1L, 2L, 3L
  x4 (Temporal Sensitivity):
    - Regression only: 1L, 2L, 3L

Total: 18 configs × 3 seeds = 54 runs
Each run trains a separate CamemBERT encoder + head on 948 silver-labelled claims.
Evaluated on 26 gold test claims (per-dim ρ against LLM silver labels).

Usage: python architecture_ablation.py
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "camembert-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 2e-5
WARMUP_RATIO = 0.1
PATIENCE = 4
DROPOUT = 0.1
NUM_RUNS = 3
MULTITASK_ALPHA = 0.3  # weight for classification loss in multi-task

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

DATA_DIR = Path(os.environ.get("SIRS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
RESULTS_DIR = Path(os.environ.get("SIRS_RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Device: {DEVICE}, AMP: {USE_AMP}")

# ─── CLASSIFICATION BINS ───────────────────────────────────────────────────
# x1: 5 sensitivity levels (matching LLM rubric)
X1_BINS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]  # 5 classes
X1_NUM_CLASSES = 5

# x3: 3 scope levels (local / regional / national)
X3_BINS = [0.0, 0.33, 0.66, 1.01]  # 3 classes
X3_NUM_CLASSES = 3


def score_to_class(score, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= score < bins[i + 1]:
            return i
    return len(bins) - 2


# ─── DATASET ────────────────────────────────────────────────────────────────
class ClaimsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item['claim'], max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'x1': torch.tensor(item['x1'], dtype=torch.float32),
            'x2': torch.tensor(item['x2'], dtype=torch.float32),
            'x3': torch.tensor(item['x3'], dtype=torch.float32),
            'x4': torch.tensor(item['x4'], dtype=torch.float32),
            'x1_class': torch.tensor(score_to_class(item['x1'], X1_BINS), dtype=torch.long),
            'x3_class': torch.tensor(score_to_class(item['x3'], X3_BINS), dtype=torch.long),
        }


# ─── MODELS ─────────────────────────────────────────────────────────────────

class RegressionExtractor(nn.Module):
    """Pure regression: CamemBERT → MLP → sigmoid → score in [0,1]."""
    def __init__(self, encoder_path, hidden_dims, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)

        layers = []
        in_dim = encoder_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        layers.extend([
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        ])
        self.head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return {'score': self.head(h).squeeze(-1)}


class MultiTaskExtractor(nn.Module):
    """Multi-task: CamemBERT → shared hidden → (classification head + regression head).
    Classification provides auxiliary signal; regression is the primary output."""
    def __init__(self, encoder_path, hidden_dims, num_classes, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)

        # Shared hidden layers
        shared_layers = []
        in_dim = encoder_dim
        for h_dim in hidden_dims:
            shared_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        self.shared = nn.Sequential(*shared_layers)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        shared_h = self.shared(h)
        return {
            'score': self.reg_head(shared_h).squeeze(-1),
            'logits': self.cls_head(shared_h),
        }


# ─── HIDDEN DIM CONFIGS PER DIMENSION ──────────────────────────────────────
# These define the MLP hidden layers for each depth variant
HIDDEN_CONFIGS = {
    'x1': {1: [256], 2: [256, 128], 3: [256, 128, 64]},
    'x2': {1: [256], 2: [256, 64],  3: [256, 64, 32]},
    'x3': {1: [128], 2: [128, 64],  3: [128, 64, 32]},
    'x4': {1: [128], 2: [128, 64],  3: [128, 64, 32]},
}


# ─── TRAINING ───────────────────────────────────────────────────────────────
def train_and_evaluate(model, train_loader, val_loader, test_loader, dim_name,
                       gold_data, is_multitask=False, cls_field=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = GradScaler('cuda', enabled=USE_AMP)

    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    best_val_rho = -1
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            target = batch[dim_name].to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda', enabled=USE_AMP):
                out = model(ids, mask)
                reg_loss = mse_fn(out['score'], target)

                if is_multitask and 'logits' in out:
                    cls_target = batch[cls_field].to(DEVICE)
                    cls_loss = ce_fn(out['logits'], cls_target)
                    loss = (1 - MULTITASK_ALPHA) * reg_loss + MULTITASK_ALPHA * cls_loss
                else:
                    loss = reg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # ── Validate ──
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast('cuda', enabled=USE_AMP):
                    out = model(ids, mask)
                all_pred.extend(out['score'].cpu().numpy().tolist())
                all_true.extend(batch[dim_name].numpy().tolist())

        val_rho = spearmanr(all_true, all_pred)[0]
        if np.isnan(val_rho):
            val_rho = 0.0

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # ── Evaluate on gold test ──
    model.load_state_dict(best_state)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            with autocast('cuda', enabled=USE_AMP):
                out = model(ids, mask)
            test_preds.extend(out['score'].cpu().numpy().tolist())

    test_trues = [item[dim_name] for item in gold_data]
    test_rho = spearmanr(test_trues, test_preds)[0]
    if np.isnan(test_rho):
        test_rho = 0.0
    test_mse = mean_squared_error(test_trues, test_preds)

    return {
        'val_rho': float(best_val_rho),
        'test_rho': float(test_rho),
        'test_mse': float(test_mse),
    }


# ─── EXPERIMENT DEFINITIONS ────────────────────────────────────────────────
EXPERIMENTS = []

# x1: regression (1L, 2L, 3L) + multi-task (1L, 2L, 3L)
for depth in [1, 2, 3]:
    EXPERIMENTS.append({
        'name': f'x1_reg_{depth}L',
        'dim': 'x1',
        'depth': depth,
        'mode': 'regression',
        'hidden_dims': HIDDEN_CONFIGS['x1'][depth],
    })
    EXPERIMENTS.append({
        'name': f'x1_mt_{depth}L',
        'dim': 'x1',
        'depth': depth,
        'mode': 'multitask',
        'hidden_dims': HIDDEN_CONFIGS['x1'][depth],
        'num_classes': X1_NUM_CLASSES,
        'cls_field': 'x1_class',
    })

# x2: regression only (1L, 2L, 3L)
for depth in [1, 2, 3]:
    EXPERIMENTS.append({
        'name': f'x2_reg_{depth}L',
        'dim': 'x2',
        'depth': depth,
        'mode': 'regression',
        'hidden_dims': HIDDEN_CONFIGS['x2'][depth],
    })

# x3: regression (1L, 2L, 3L) + multi-task (1L, 2L, 3L)
for depth in [1, 2, 3]:
    EXPERIMENTS.append({
        'name': f'x3_reg_{depth}L',
        'dim': 'x3',
        'depth': depth,
        'mode': 'regression',
        'hidden_dims': HIDDEN_CONFIGS['x3'][depth],
    })
    EXPERIMENTS.append({
        'name': f'x3_mt_{depth}L',
        'dim': 'x3',
        'depth': depth,
        'mode': 'multitask',
        'hidden_dims': HIDDEN_CONFIGS['x3'][depth],
        'num_classes': X3_NUM_CLASSES,
        'cls_field': 'x3_class',
    })

# x4: regression only (1L, 2L, 3L)
for depth in [1, 2, 3]:
    EXPERIMENTS.append({
        'name': f'x4_reg_{depth}L',
        'dim': 'x4',
        'depth': depth,
        'mode': 'regression',
        'hidden_dims': HIDDEN_CONFIGS['x4'][depth],
    })


# ─── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("ARCHITECTURE ABLATION STUDY")
    logger.info(f"  {len(EXPERIMENTS)} configs × {NUM_RUNS} seeds = {len(EXPERIMENTS) * NUM_RUNS} total runs")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    with open(DATA_DIR / "training_data_with_dimensions.json") as f:
        all_train_data = json.load(f)
    with open(DATA_DIR / "gold_test_with_dimensions.json") as f:
        gold_data_raw = json.load(f)

    gold_data = []
    for item in gold_data_raw:
        gold_data.append({
            'claim': item['claim'],
            'risk_score': item['risk_score'],
            'x1': item['llm_x1'],
            'x2': item['llm_x2'],
            'x3': item['llm_x3'],
            'x4': item['llm_x4'],
        })

    logger.info(f"Training: {len(all_train_data)} claims, Gold test: {len(gold_data)} claims")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_results = {}

    for exp in EXPERIMENTS:
        exp_name = exp['name']
        run_results = []

        for run in range(NUM_RUNS):
            seed = SEED + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            shuffled = all_train_data.copy()
            random.shuffle(shuffled)
            split = int(0.85 * len(shuffled))
            train_split = shuffled[:split]
            val_split = shuffled[split:]

            train_loader = DataLoader(
                ClaimsDataset(train_split, tokenizer, MAX_LEN),
                batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = DataLoader(
                ClaimsDataset(val_split, tokenizer, MAX_LEN),
                batch_size=BATCH_SIZE, num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                ClaimsDataset(gold_data, tokenizer, MAX_LEN),
                batch_size=len(gold_data), num_workers=0, pin_memory=True
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment: {exp_name}, Run {run+1}/{NUM_RUNS} (seed={seed})")
            logger.info(f"  Mode: {exp['mode']}, Hidden: {exp['hidden_dims']}")
            logger.info(f"{'='*60}")

            # Create model
            if exp['mode'] == 'regression':
                model = RegressionExtractor(
                    MODEL_NAME, hidden_dims=exp['hidden_dims'], dropout=DROPOUT
                ).to(DEVICE)
                is_mt = False
                cls_field = None
            else:
                model = MultiTaskExtractor(
                    MODEL_NAME, hidden_dims=exp['hidden_dims'],
                    num_classes=exp['num_classes'], dropout=DROPOUT
                ).to(DEVICE)
                is_mt = True
                cls_field = exp['cls_field']

            n_head_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' not in n)
            logger.info(f"  Head parameters: {n_head_params:,}")

            result = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                exp['dim'], gold_data, is_multitask=is_mt, cls_field=cls_field
            )
            run_results.append(result)

            logger.info(f"  Result: val_ρ={result['val_rho']:.3f}, test_ρ={result['test_rho']:.3f}, MSE={result['test_mse']:.4f}")

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Aggregate
        avg_rho = np.mean([r['test_rho'] for r in run_results])
        std_rho = np.std([r['test_rho'] for r in run_results])
        avg_mse = np.mean([r['test_mse'] for r in run_results])

        all_results[exp_name] = {
            'dim': exp['dim'],
            'mode': exp['mode'],
            'depth': exp['depth'],
            'hidden_dims': exp['hidden_dims'],
            'avg_test_rho': float(avg_rho),
            'std_test_rho': float(std_rho),
            'avg_test_mse': float(avg_mse),
            'runs': run_results,
        }

        logger.info(f"\n>>> {exp_name}: avg ρ = {avg_rho:.3f} ± {std_rho:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("ARCHITECTURE ABLATION RESULTS (averaged over %d runs)", NUM_RUNS)
    logger.info("=" * 80)

    # Table 1: x1 - Regression vs Multi-task
    logger.info("\n--- x1 (Topic Sensitivity): Regression vs Multi-task ---")
    logger.info(f"  {'Config':<20s} {'1 layer':>14s} {'2 layers':>14s} {'3 layers':>14s}")
    logger.info(f"  {'-'*62}")
    for mode, label in [('reg', 'Regression'), ('mt', 'Multi-task')]:
        row = f"  {label:<20s}"
        for depth in [1, 2, 3]:
            key = f"x1_{mode}_{depth}L"
            r = all_results[key]
            row += f" {r['avg_test_rho']:.3f}±{r['std_test_rho']:.3f}".rjust(14)
        logger.info(row)

    # Table 2: x2 - Depth only
    logger.info("\n--- x2 (Actionability): Depth ablation ---")
    row = f"  {'Regression':<20s}"
    for depth in [1, 2, 3]:
        key = f"x2_reg_{depth}L"
        r = all_results[key]
        row += f" {r['avg_test_rho']:.3f}±{r['std_test_rho']:.3f}".rjust(14)
    logger.info(f"  {'Config':<20s} {'1 layer':>14s} {'2 layers':>14s} {'3 layers':>14s}")
    logger.info(f"  {'-'*62}")
    logger.info(row)

    # Table 3: x3 - Regression vs Multi-task
    logger.info("\n--- x3 (Audience Scope): Regression vs Multi-task ---")
    logger.info(f"  {'Config':<20s} {'1 layer':>14s} {'2 layers':>14s} {'3 layers':>14s}")
    logger.info(f"  {'-'*62}")
    for mode, label in [('reg', 'Regression'), ('mt', 'Multi-task')]:
        row = f"  {label:<20s}"
        for depth in [1, 2, 3]:
            key = f"x3_{mode}_{depth}L"
            r = all_results[key]
            row += f" {r['avg_test_rho']:.3f}±{r['std_test_rho']:.3f}".rjust(14)
        logger.info(row)

    # Table 4: x4 - Depth only
    logger.info("\n--- x4 (Temporal Sensitivity): Depth ablation ---")
    row = f"  {'Regression':<20s}"
    for depth in [1, 2, 3]:
        key = f"x4_reg_{depth}L"
        r = all_results[key]
        row += f" {r['avg_test_rho']:.3f}±{r['std_test_rho']:.3f}".rjust(14)
    logger.info(f"  {'Config':<20s} {'1 layer':>14s} {'2 layers':>14s} {'3 layers':>14s}")
    logger.info(f"  {'-'*62}")
    logger.info(row)

    # Best config per dimension
    logger.info("\n--- BEST CONFIGURATION PER DIMENSION ---")
    for dim in ['x1', 'x2', 'x3', 'x4']:
        dim_results = {k: v for k, v in all_results.items() if v['dim'] == dim}
        # Selection on the VALIDATION split only (mean val rho over seeds); test rho is reported, never used to choose.
        best_key = max(dim_results, key=lambda k: sum(r['val_rho'] for r in dim_results[k]['runs']) / len(dim_results[k]['runs']))
        best = dim_results[best_key]
        logger.info(f"  {dim}: {best_key} (ρ={best['avg_test_rho']:.3f}±{best['std_test_rho']:.3f}, hidden={best['hidden_dims']})")

    logger.info(f"\nTotal time: {total_time/60:.1f} minutes")

    # Save results
    results_file = RESULTS_DIR / "architecture_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")
