"""
Training script with ablation-validated architectures.
Trains both separate (4 encoders) and unified (1 encoder, 4 heads) models.
All dimensions use pure regression with ablation-optimal depths:
  x1: 3L [256, 128, 64]
  x2: 1L [256]
  x3: 1L [128]
  x4: 1L [128]

Evaluates end-to-end risk score through SIRS formula.
"""

import json
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

DATA_DIR = Path("/home/oussama.sarsar/paper2/data/final")
RESULTS_DIR = Path("/home/oussama.sarsar/paper2/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# SIRS formula weights
SIRS_WEIGHTS = [0.30, 0.25, 0.25, 0.20]  # x1, x2, x3, x4

# Ablation-validated architectures (all pure regression)
SEPARATE_CONFIGS = {
    'x1': [256, 128, 64],  # 3L
    'x2': [256],            # 1L
    'x3': [128],            # 1L
    'x4': [128],            # 1L
}

UNIFIED_CONFIGS = {
    'x1': [256, 128, 64],  # 3L
    'x2': [256],            # 1L
    'x3': [128],            # 1L
    'x4': [128],            # 1L
}


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
        }


# ─── MODELS ─────────────────────────────────────────────────────────────────
def build_head(encoder_dim, hidden_dims, dropout):
    layers = []
    in_dim = encoder_dim
    for h_dim in hidden_dims:
        layers.extend([nn.Dropout(dropout), nn.Linear(in_dim, h_dim), nn.ReLU()])
        in_dim = h_dim
    layers.extend([nn.Dropout(dropout), nn.Linear(in_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


class SeparateExtractor(nn.Module):
    def __init__(self, encoder_path, hidden_dims, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.head = build_head(encoder_dim, hidden_dims, dropout)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return self.head(h).squeeze(-1)


class UnifiedExtractor(nn.Module):
    def __init__(self, encoder_path, head_configs, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.heads = nn.ModuleDict({
            dim: build_head(encoder_dim, hdims, dropout)
            for dim, hdims in head_configs.items()
        })

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return {dim: head(h).squeeze(-1) for dim, head in self.heads.items()}


# ─── SIRS FORMULA ───────────────────────────────────────────────────────────
def sirs_score(x1, x2, x3, x4):
    return min(1.0, 0.30 * x1 + 0.25 * x2 + 0.25 * x3 + 0.20 * x4)


# ─── TRAINING ───────────────────────────────────────────────────────────────
def train_separate(model, train_loader, val_loader, dim_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    mse_fn = nn.MSELoss()

    best_val_rho = -1
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            target = batch[dim_name].to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda', enabled=USE_AMP):
                pred = model(ids, mask)
                loss = mse_fn(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast('cuda', enabled=USE_AMP):
                    pred = model(ids, mask)
                preds.extend(pred.cpu().numpy().tolist())
                trues.extend(batch[dim_name].numpy().tolist())

        val_rho = spearmanr(trues, preds)[0]
        if np.isnan(val_rho): val_rho = 0.0

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break

    model.load_state_dict(best_state)
    return model


def train_unified(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    mse_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda', enabled=USE_AMP):
                out = model(ids, mask)
                loss = sum(mse_fn(out[d], batch[d].to(DEVICE)) for d in ['x1','x2','x3','x4'])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast('cuda', enabled=USE_AMP):
                    out = model(ids, mask)
                val_loss += sum(mse_fn(out[d], batch[d].to(DEVICE)).item() for d in ['x1','x2','x3','x4'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break

    model.load_state_dict(best_state)
    return model


# ─── EVALUATION ─────────────────────────────────────────────────────────────
def evaluate_separate_models(models, test_loader, gold_data):
    """Evaluate 4 separate models on gold test, compute per-dim and end-to-end metrics."""
    dim_preds = {d: [] for d in ['x1','x2','x3','x4']}

    for dim, model in models.items():
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast('cuda', enabled=USE_AMP):
                    pred = model(ids, mask)
                dim_preds[dim].extend(pred.cpu().numpy().tolist())

    results = {}
    for dim in ['x1','x2','x3','x4']:
        trues = [item[dim] for item in gold_data]
        rho = spearmanr(trues, dim_preds[dim])[0]
        if np.isnan(rho): rho = 0.0
        mse = mean_squared_error(trues, dim_preds[dim])
        results[dim] = {'rho': float(rho), 'mse': float(mse)}

    # End-to-end risk score
    pred_risks = [sirs_score(dim_preds['x1'][i], dim_preds['x2'][i], dim_preds['x3'][i], dim_preds['x4'][i])
                  for i in range(len(gold_data))]
    true_risks = [item['risk_score'] for item in gold_data]
    e2e_rho = spearmanr(true_risks, pred_risks)[0]
    if np.isnan(e2e_rho): e2e_rho = 0.0
    e2e_mse = mean_squared_error(true_risks, pred_risks)
    e2e_mae = float(np.mean(np.abs(np.array(true_risks) - np.array(pred_risks))))
    results['end_to_end'] = {'rho': float(e2e_rho), 'mse': float(e2e_mse), 'mae': float(e2e_mae)}
    return results


def evaluate_unified_model(model, test_loader, gold_data):
    """Evaluate unified model on gold test."""
    model.eval()
    dim_preds = {d: [] for d in ['x1','x2','x3','x4']}

    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            with autocast('cuda', enabled=USE_AMP):
                out = model(ids, mask)
            for dim in ['x1','x2','x3','x4']:
                dim_preds[dim].extend(out[dim].cpu().numpy().tolist())

    results = {}
    for dim in ['x1','x2','x3','x4']:
        trues = [item[dim] for item in gold_data]
        rho = spearmanr(trues, dim_preds[dim])[0]
        if np.isnan(rho): rho = 0.0
        mse = mean_squared_error(trues, dim_preds[dim])
        results[dim] = {'rho': float(rho), 'mse': float(mse)}

    pred_risks = [sirs_score(dim_preds['x1'][i], dim_preds['x2'][i], dim_preds['x3'][i], dim_preds['x4'][i])
                  for i in range(len(gold_data))]
    true_risks = [item['risk_score'] for item in gold_data]
    e2e_rho = spearmanr(true_risks, pred_risks)[0]
    if np.isnan(e2e_rho): e2e_rho = 0.0
    e2e_mse = mean_squared_error(true_risks, pred_risks)
    e2e_mae = float(np.mean(np.abs(np.array(true_risks) - np.array(pred_risks))))
    results['end_to_end'] = {'rho': float(e2e_rho), 'mse': float(e2e_mse), 'mae': float(e2e_mae)}
    return results


# ─── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info(f"Training: {len(all_train_data)}, Gold: {len(gold_data)}, Device: {DEVICE}")

    all_separate_results = []
    all_unified_results = []

    for run in range(NUM_RUNS):
        seed = SEED + run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        shuffled = all_train_data.copy()
        random.shuffle(shuffled)
        split = int(0.85 * len(shuffled))
        train_split, val_split = shuffled[:split], shuffled[split:]

        train_loader = DataLoader(ClaimsDataset(train_split, tokenizer, MAX_LEN),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(ClaimsDataset(val_split, tokenizer, MAX_LEN),
                                batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
        test_loader = DataLoader(ClaimsDataset(gold_data, tokenizer, MAX_LEN),
                                 batch_size=len(gold_data), num_workers=0, pin_memory=True)

        # ── SEPARATE MODELS ──
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run+1}/{NUM_RUNS} (seed={seed}) - SEPARATE MODELS")
        logger.info(f"{'='*60}")

        sep_models = {}
        for dim in ['x1', 'x2', 'x3', 'x4']:
            logger.info(f"  Training {dim} separate (hidden={SEPARATE_CONFIGS[dim]})...")
            model = SeparateExtractor(MODEL_NAME, SEPARATE_CONFIGS[dim], dropout=DROPOUT).to(DEVICE)
            model = train_separate(model, train_loader, val_loader, dim)
            sep_models[dim] = model

        sep_results = evaluate_separate_models(sep_models, test_loader, gold_data)
        all_separate_results.append(sep_results)
        logger.info(f"  Separate results:")
        for dim in ['x1','x2','x3','x4']:
            logger.info(f"    {dim}: ρ={sep_results[dim]['rho']:.3f}, MSE={sep_results[dim]['mse']:.4f}")
        logger.info(f"    End-to-end: ρ={sep_results['end_to_end']['rho']:.3f}")

        del sep_models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── UNIFIED MODEL ──
        logger.info(f"\n  Run {run+1}/{NUM_RUNS} - UNIFIED MODEL")
        unified = UnifiedExtractor(MODEL_NAME, UNIFIED_CONFIGS, dropout=DROPOUT).to(DEVICE)
        unified = train_unified(unified, train_loader, val_loader)

        uni_results = evaluate_unified_model(unified, test_loader, gold_data)
        all_unified_results.append(uni_results)
        logger.info(f"  Unified results:")
        for dim in ['x1','x2','x3','x4']:
            logger.info(f"    {dim}: ρ={uni_results[dim]['rho']:.3f}, MSE={uni_results[dim]['mse']:.4f}")
        logger.info(f"    End-to-end: ρ={uni_results['end_to_end']['rho']:.3f}")

        del unified
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATE AND REPORT
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS (averaged over %d runs)", NUM_RUNS)
    logger.info("=" * 70)

    # Average separate results
    logger.info("\n--- SEPARATE MODELS (4 encoders) ---")
    for dim in ['x1','x2','x3','x4','end_to_end']:
        rhos = [r[dim]['rho'] for r in all_separate_results]
        mses = [r[dim]['mse'] for r in all_separate_results]
        label = dim if dim != 'end_to_end' else 'E2E risk'
        logger.info(f"  {label}: ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}, MSE={np.mean(mses):.4f}")

    # Average unified results
    logger.info("\n--- UNIFIED MODEL (1 encoder, 4 heads) ---")
    for dim in ['x1','x2','x3','x4','end_to_end']:
        rhos = [r[dim]['rho'] for r in all_unified_results]
        mses = [r[dim]['mse'] for r in all_unified_results]
        label = dim if dim != 'end_to_end' else 'E2E risk'
        logger.info(f"  {label}: ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}, MSE={np.mean(mses):.4f}")

    # E2E comparison table
    logger.info("\n--- END-TO-END COMPARISON ---")
    sep_e2e = [r['end_to_end'] for r in all_separate_results]
    uni_e2e = [r['end_to_end'] for r in all_unified_results]
    logger.info(f"  Separate: ρ={np.mean([r['rho'] for r in sep_e2e]):.3f}, MSE={np.mean([r['mse'] for r in sep_e2e]):.4f}, MAE={np.mean([r['mae'] for r in sep_e2e]):.3f}")
    logger.info(f"  Unified:  ρ={np.mean([r['rho'] for r in uni_e2e]):.3f}, MSE={np.mean([r['mse'] for r in uni_e2e]):.4f}, MAE={np.mean([r['mae'] for r in uni_e2e]):.3f}")

    logger.info(f"\nTotal time: {total_time/60:.1f} minutes")

    # Save results
    output = {
        'separate_runs': all_separate_results,
        'unified_runs': all_unified_results,
        'config': {
            'separate_heads': SEPARATE_CONFIGS,
            'unified_heads': UNIFIED_CONFIGS,
            'num_runs': NUM_RUNS,
            'seeds': [SEED + i for i in range(NUM_RUNS)],
        }
    }
    results_file = RESULTS_DIR / "corrected_results.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")
