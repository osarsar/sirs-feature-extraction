"""
CamemBERT-based extractors for SIRS dimensions (x1-x4).
Trains on real Moroccan claims with LLM silver labels.
Supports: separate (4 independent models) and unified (1 shared encoder, 4 heads).
Evaluates end-to-end: extract x1-x4 → SIRS formula → risk_score → compare with gold.

Usage:
    cd ~/Desktop/paper2
    python3 scripts/train_camembert_extractors.py
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "camembert-base"
DAPT_MODEL_PATH = Path(__file__).parent.parent / "models" / "camembert_dapt" / "final"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 2e-5
WARMUP_RATIO = 0.1
PATIENCE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

DATA_DIR = Path(__file__).parent.parent / "data" / "final"
OUTPUT_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logger.info(f"Device: {DEVICE}, AMP: {USE_AMP}")

# ─── SIRS FORMULA (from Paper 1) ───────────────────────────────────────────
# Weights from Paper 1: w1=0.30, w2=0.25, w3=0.25, w4=0.20
SIRS_WEIGHTS = [0.30, 0.25, 0.25, 0.20]

def sirs_formula(x1, x2, x3, x4):
    """Paper 1 base formula: weighted sum capped at 1.0"""
    return min(1.0, SIRS_WEIGHTS[0]*x1 + SIRS_WEIGHTS[1]*x2 + SIRS_WEIGHTS[2]*x3 + SIRS_WEIGHTS[3]*x4)


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
            'risk_score': torch.tensor(item.get('risk_score', 0.5), dtype=torch.float32),
        }


# ─── MODELS ─────────────────────────────────────────────────────────────────
class SeparateExtractor(nn.Module):
    """Single-dimension extractor: CamemBERT + regression head."""
    def __init__(self, encoder_path, dim_name, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.dim_name = dim_name

        if dim_name == 'x2':
            # Deeper head for actionability (pragmatic signals)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(encoder_dim, 256), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(encoder_dim, 256), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(256, 1), nn.Sigmoid(),
            )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return self.head(h).squeeze(-1)


class UnifiedExtractor(nn.Module):
    """Shared encoder, 4 regression heads."""
    def __init__(self, encoder_path, encoder_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)

        self.x1_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 1), nn.Sigmoid(),
        )
        self.x2_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.x3_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.x4_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return {
            'x1': self.x1_head(h).squeeze(-1),
            'x2': self.x2_head(h).squeeze(-1),
            'x3': self.x3_head(h).squeeze(-1),
            'x4': self.x4_head(h).squeeze(-1),
        }


# ─── TRAINING FUNCTIONS ────────────────────────────────────────────────────
def train_separate(model, train_loader, val_loader, dim_name, optimizer, scheduler, scaler, epochs, patience):
    """Train a single-dimension extractor."""
    best_rho = -1
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            target = batch[dim_name].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                pred = model(ids, mask)
                loss = nn.MSELoss()(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast(enabled=USE_AMP):
                    pred = model(ids, mask)
                all_pred.extend(pred.cpu().numpy().tolist())
                all_true.extend(batch[dim_name].numpy().tolist())

        rho = spearmanr(all_true, all_pred)[0]
        mse = mean_squared_error(all_true, all_pred)
        avg_loss = total_loss / len(train_loader)
        logger.info(f"  [{dim_name}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, ρ: {rho:.3f}, MSE: {mse:.4f}")

        if rho > best_rho:
            best_rho = rho
            wait = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"camembert_separate_{dim_name}_best.pt")
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"  [{dim_name}] Early stopping at epoch {epoch+1}")
                break

    return best_rho


def train_unified(model, train_loader, val_loader, optimizer, scheduler, scaler, epochs, patience):
    """Train the unified 4-head model."""
    best_avg_rho = -1
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                preds = model(ids, mask)
                loss = sum(
                    nn.MSELoss()(preds[dim], batch[dim].to(DEVICE))
                    for dim in ['x1', 'x2', 'x3', 'x4']
                ) / 4.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate per dimension
        model.eval()
        dim_preds = {d: [] for d in ['x1', 'x2', 'x3', 'x4']}
        dim_trues = {d: [] for d in ['x1', 'x2', 'x3', 'x4']}
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast(enabled=USE_AMP):
                    preds = model(ids, mask)
                for dim in ['x1', 'x2', 'x3', 'x4']:
                    dim_preds[dim].extend(preds[dim].cpu().numpy().tolist())
                    dim_trues[dim].extend(batch[dim].numpy().tolist())

        rhos = {}
        for dim in ['x1', 'x2', 'x3', 'x4']:
            rhos[dim] = spearmanr(dim_trues[dim], dim_preds[dim])[0]

        avg_rho = np.mean(list(rhos.values()))
        avg_loss = total_loss / len(train_loader)
        logger.info(f"  [Unified] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                     f"x1:{rhos['x1']:.3f} x2:{rhos['x2']:.3f} x3:{rhos['x3']:.3f} x4:{rhos['x4']:.3f} avg:{avg_rho:.3f}")

        if avg_rho > best_avg_rho:
            best_avg_rho = avg_rho
            wait = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "camembert_unified_best.pt")
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"  [Unified] Early stopping at epoch {epoch+1}")
                break

    return best_avg_rho


# ─── EVALUATION ─────────────────────────────────────────────────────────────
def evaluate_end_to_end(predictions_x1x4, gold_data):
    """
    Compute end-to-end risk score using SIRS formula and compare with gold.
    predictions_x1x4: list of (x1, x2, x3, x4) tuples
    gold_data: list of dicts with 'risk_score' (human median)
    """
    pred_risks = []
    true_risks = []
    for (x1, x2, x3, x4), item in zip(predictions_x1x4, gold_data):
        pred_risk = sirs_formula(x1, x2, x3, x4)
        pred_risks.append(pred_risk)
        true_risks.append(item['risk_score'])

    rho, p = spearmanr(true_risks, pred_risks)
    mse = mean_squared_error(true_risks, pred_risks)
    mae = mean_absolute_error(true_risks, pred_risks)
    return {'rho': rho, 'p': p, 'mse': mse, 'mae': mae}


def predict_dimensions(model, loader, mode='separate', dim_name=None):
    """Get predictions from a model."""
    model.eval()
    if mode == 'separate':
        preds = []
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast(enabled=USE_AMP):
                    pred = model(ids, mask)
                preds.extend(pred.cpu().numpy().tolist())
        return preds
    else:  # unified
        dim_preds = {d: [] for d in ['x1', 'x2', 'x3', 'x4']}
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                with autocast(enabled=USE_AMP):
                    out = model(ids, mask)
                for d in ['x1', 'x2', 'x3', 'x4']:
                    dim_preds[d].extend(out[d].cpu().numpy().tolist())
        return dim_preds


# ─── RULE-BASED BASELINE ───────────────────────────────────────────────────
TOPIC_SENSITIVITY_LOOKUP = {
    "roi": 0.90, "sahara": 0.95, "autonomie": 0.90, "parti": 0.75,
    "élection": 0.80, "protestation": 0.76, "grève": 0.71, "syndicat": 0.67,
    "religion": 0.57, "islamist": 0.52, "santé": 0.48, "éducation": 0.43,
    "normalisation": 0.33, "palestin": 0.29, "gouvernement": 0.60,
    "agricol": 0.19, "transport": 0.14, "coupe": 0.05,
}

ACTION_KEYWORDS = {
    1.0: ["boycott", "grève générale", "mobilisation", "ne vaccinez pas", "refusez"],
    0.8: ["attaque", "épidémie", "catastrophe", "mort", "tués", "pandémie", "séisme"],
    0.6: ["méfiez-vous", "attention", "protégez-vous", "faites attention"],
    0.5: ["scandale", "corruption", "conspiration", "honte", "inacceptable"],
}

NATIONAL_MARKERS = ["maroc", "gouvernement", "parlement", "roi", "royaume", "national", "ministère"]
REGIONAL_MARKERS = ["casablanca", "rabat", "marrakech", "fès", "tanger", "agadir", "région"]
LOCAL_MARKERS = ["commune", "quartier", "douar", "village", "local"]

TEMPORAL_KEYWORDS = {
    1.0: ["crise", "urgence", "séisme", "pandémie", "attaque", "état d'urgence"],
    0.9: ["élections", "scrutin", "vote", "campagne électorale"],
    0.7: ["ramadan", "aïd", "mosquée", "pèlerinage"],
    0.6: ["dernière heure", "imminent", "urgent", "alerte"],
}

def rule_based_extract(claim):
    """Paper 1 keyword-based extraction."""
    text = claim.lower()

    # x1
    x1 = 0.15
    for kw, score in TOPIC_SENSITIVITY_LOOKUP.items():
        if kw in text and score > x1:
            x1 = score

    # x2
    x2_total = 0
    for weight, keywords in ACTION_KEYWORDS.items():
        if any(k in text for k in keywords):
            x2_total = max(x2_total, weight * 0.5)
    x2 = min(1.0, x2_total)

    # x3
    n_nat = sum(1 for m in NATIONAL_MARKERS if m in text)
    n_reg = sum(1 for m in REGIONAL_MARKERS if m in text)
    n_loc = sum(1 for m in LOCAL_MARKERS if m in text)
    if n_nat >= 2: x3 = 0.85
    elif n_reg >= 1: x3 = 0.50
    elif n_loc >= 1: x3 = 0.20
    else: x3 = 0.30

    # x4
    x4 = 0.10
    for weight, keywords in TEMPORAL_KEYWORDS.items():
        if any(k in text for k in keywords):
            x4 = max(x4, weight * 0.6)

    return x1, x2, x3, x4


# ─── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()

    # Load data
    logger.info("Loading data...")
    with open(DATA_DIR / "training_data_with_dimensions.json") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "gold_test_with_dimensions.json") as f:
        gold_data = json.load(f)

    # Prepare gold test (use LLM dimension scores as features for LLM baseline)
    gold_for_eval = []
    for item in gold_data:
        gold_for_eval.append({
            'claim': item['claim'],
            'risk_score': item['risk_score'],
            'x1': item['llm_x1'],
            'x2': item['llm_x2'],
            'x3': item['llm_x3'],
            'x4': item['llm_x4'],
        })

    # Split training: 85% train, 15% val
    random.shuffle(train_data)
    split = int(0.85 * len(train_data))
    train_split = train_data[:split]
    val_split = train_data[split:]

    logger.info(f"Train: {len(train_split)}, Val: {len(val_split)}, Gold test: {len(gold_for_eval)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(ClaimsDataset(train_split, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ClaimsDataset(val_split, tokenizer, MAX_LEN),
                            batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ClaimsDataset(gold_for_eval, tokenizer, MAX_LEN),
                             batch_size=len(gold_for_eval), num_workers=4, pin_memory=True)

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 1. RULE-BASED BASELINE
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("EVALUATING: Rule-based baseline (Paper 1)")
    logger.info("="*60)

    rule_preds = [rule_based_extract(item['claim']) for item in gold_for_eval]
    rule_results = evaluate_end_to_end(rule_preds, gold_for_eval)
    logger.info(f"Rule-based end-to-end: ρ={rule_results['rho']:.3f}, MSE={rule_results['mse']:.4f}")
    all_results['rule_based'] = rule_results

    # Per-dimension rule-based
    for i, dim in enumerate(['x1', 'x2', 'x3', 'x4']):
        preds = [p[i] for p in rule_preds]
        trues = [item[dim] for item in gold_for_eval]
        rho = spearmanr(trues, preds)[0]
        logger.info(f"  Rule-based {dim}: ρ={rho:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. LLM ZERO-SHOT BASELINE
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("EVALUATING: LLM zero-shot (Claude Opus)")
    logger.info("="*60)

    llm_preds = [(item['llm_x1'], item['llm_x2'], item['llm_x3'], item['llm_x4']) for item in gold_data]
    llm_results = evaluate_end_to_end(llm_preds, gold_for_eval)
    logger.info(f"LLM end-to-end: ρ={llm_results['rho']:.3f}, MSE={llm_results['mse']:.4f}")
    all_results['llm_zero_shot'] = llm_results

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SEPARATE EXTRACTORS (vanilla CamemBERT)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("TRAINING: 4 separate CamemBERT extractors (vanilla)")
    logger.info("="*60)

    separate_preds_vanilla = {}
    for dim in ['x1', 'x2', 'x3', 'x4']:
        logger.info(f"\n--- Training {dim} extractor (vanilla) ---")
        model = SeparateExtractor(MODEL_NAME, dim).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
        scaler = GradScaler(enabled=USE_AMP)

        best_rho = train_separate(model, train_loader, val_loader, dim, optimizer, scheduler, scaler, EPOCHS, PATIENCE)
        logger.info(f"  Best val ρ for {dim}: {best_rho:.3f}")

        # Load best and predict on gold
        model.load_state_dict(torch.load(OUTPUT_DIR / f"camembert_separate_{dim}_best.pt", weights_only=True))
        preds = predict_dimensions(model, test_loader, mode='separate')
        separate_preds_vanilla[dim] = preds
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # End-to-end for separate vanilla
    sep_v_tuples = list(zip(separate_preds_vanilla['x1'], separate_preds_vanilla['x2'],
                             separate_preds_vanilla['x3'], separate_preds_vanilla['x4']))
    sep_v_results = evaluate_end_to_end(sep_v_tuples, gold_for_eval)
    logger.info(f"\nSeparate (vanilla) end-to-end: ρ={sep_v_results['rho']:.3f}")
    all_results['separate_vanilla'] = sep_v_results

    # Per-dimension separate vanilla
    for dim in ['x1', 'x2', 'x3', 'x4']:
        trues = [item[dim] for item in gold_for_eval]
        rho = spearmanr(trues, separate_preds_vanilla[dim])[0]
        mse = mean_squared_error(trues, separate_preds_vanilla[dim])
        all_results[f'separate_vanilla_{dim}'] = {'rho': rho, 'mse': mse}
        logger.info(f"  Separate vanilla {dim}: ρ={rho:.3f}, MSE={mse:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. UNIFIED EXTRACTOR (vanilla CamemBERT)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Unified CamemBERT extractor (vanilla)")
    logger.info("="*60)

    model = UnifiedExtractor(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = GradScaler(enabled=USE_AMP)

    best_avg = train_unified(model, train_loader, val_loader, optimizer, scheduler, scaler, EPOCHS, PATIENCE)

    # Load best and predict
    model.load_state_dict(torch.load(OUTPUT_DIR / "camembert_unified_best.pt", weights_only=True))
    uni_preds = predict_dimensions(model, test_loader, mode='unified')

    uni_tuples = list(zip(uni_preds['x1'], uni_preds['x2'], uni_preds['x3'], uni_preds['x4']))
    uni_results = evaluate_end_to_end(uni_tuples, gold_for_eval)
    logger.info(f"\nUnified (vanilla) end-to-end: ρ={uni_results['rho']:.3f}")
    all_results['unified_vanilla'] = uni_results

    for dim in ['x1', 'x2', 'x3', 'x4']:
        trues = [item[dim] for item in gold_for_eval]
        rho = spearmanr(trues, uni_preds[dim])[0]
        mse = mean_squared_error(trues, uni_preds[dim])
        all_results[f'unified_vanilla_{dim}'] = {'rho': rho, 'mse': mse}
        logger.info(f"  Unified vanilla {dim}: ρ={rho:.3f}, MSE={mse:.4f}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══════════════════════════════════════════════════════════════════════
    # 5. SEPARATE + UNIFIED with DAPT (if available)
    # ═══════════════════════════════════════════════════════════════════════
    if DAPT_MODEL_PATH.exists():
        logger.info("\n" + "="*60)
        logger.info("TRAINING: Separate + Unified with DAPT encoder")
        logger.info("="*60)

        dapt_path = str(DAPT_MODEL_PATH)

        # Separate DAPT
        separate_preds_dapt = {}
        for dim in ['x1', 'x2', 'x3', 'x4']:
            logger.info(f"\n--- Training {dim} extractor (DAPT) ---")
            model = SeparateExtractor(dapt_path, dim).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
            scaler = GradScaler(enabled=USE_AMP)

            best_rho = train_separate(model, train_loader, val_loader, dim, optimizer, scheduler, scaler, EPOCHS, PATIENCE)

            model.load_state_dict(torch.load(OUTPUT_DIR / f"camembert_separate_{dim}_best.pt", weights_only=True))
            preds = predict_dimensions(model, test_loader, mode='separate')
            separate_preds_dapt[dim] = preds
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        sep_d_tuples = list(zip(separate_preds_dapt['x1'], separate_preds_dapt['x2'],
                                 separate_preds_dapt['x3'], separate_preds_dapt['x4']))
        sep_d_results = evaluate_end_to_end(sep_d_tuples, gold_for_eval)
        logger.info(f"\nSeparate (DAPT) end-to-end: ρ={sep_d_results['rho']:.3f}")
        all_results['separate_dapt'] = sep_d_results

        for dim in ['x1', 'x2', 'x3', 'x4']:
            trues = [item[dim] for item in gold_for_eval]
            rho = spearmanr(trues, separate_preds_dapt[dim])[0]
            mse = mean_squared_error(trues, separate_preds_dapt[dim])
            all_results[f'separate_dapt_{dim}'] = {'rho': rho, 'mse': mse}

        # Unified DAPT
        logger.info(f"\n--- Training unified extractor (DAPT) ---")
        model = UnifiedExtractor(dapt_path).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
        scaler = GradScaler(enabled=USE_AMP)

        train_unified(model, train_loader, val_loader, optimizer, scheduler, scaler, EPOCHS, PATIENCE)

        model.load_state_dict(torch.load(OUTPUT_DIR / "camembert_unified_best.pt", weights_only=True))
        uni_d_preds = predict_dimensions(model, test_loader, mode='unified')

        uni_d_tuples = list(zip(uni_d_preds['x1'], uni_d_preds['x2'], uni_d_preds['x3'], uni_d_preds['x4']))
        uni_d_results = evaluate_end_to_end(uni_d_tuples, gold_for_eval)
        logger.info(f"\nUnified (DAPT) end-to-end: ρ={uni_d_results['rho']:.3f}")
        all_results['unified_dapt'] = uni_d_results

        for dim in ['x1', 'x2', 'x3', 'x4']:
            trues = [item[dim] for item in gold_for_eval]
            rho = spearmanr(trues, uni_d_preds[dim])[0]
            mse = mean_squared_error(trues, uni_d_preds[dim])
            all_results[f'unified_dapt_{dim}'] = {'rho': rho, 'mse': mse}

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.info("DAPT model not found, skipping DAPT experiments.")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*60)

    # Table 1: End-to-end comparison
    logger.info("\nTable 1: End-to-end risk ρ (26 gold claims)")
    logger.info(f"  {'Method':<35s} {'ρ':>8s} {'MSE':>8s} {'MAE':>8s}")
    logger.info(f"  {'-'*60}")
    for name, key in [
        ("Rule-based (Paper 1)", "rule_based"),
        ("Separate (vanilla)", "separate_vanilla"),
        ("Unified (vanilla)", "unified_vanilla"),
        ("Separate (DAPT)", "separate_dapt"),
        ("Unified (DAPT)", "unified_dapt"),
        ("LLM zero-shot (Claude)", "llm_zero_shot"),
    ]:
        if key in all_results:
            r = all_results[key]
            logger.info(f"  {name:<35s} {r['rho']:>8.3f} {r['mse']:>8.4f} {r['mae']:>8.4f}")

    # Table 2: Per-dimension comparison (separate vs unified, vanilla)
    logger.info("\nTable 2: Per-dimension ρ (separate vs unified, vanilla CamemBERT)")
    logger.info(f"  {'Dim':<5s} {'Sep ρ':>8s} {'Sep MSE':>8s} {'Uni ρ':>8s} {'Uni MSE':>8s}")
    logger.info(f"  {'-'*40}")
    for dim in ['x1', 'x2', 'x3', 'x4']:
        sk = f'separate_vanilla_{dim}'
        uk = f'unified_vanilla_{dim}'
        if sk in all_results and uk in all_results:
            s = all_results[sk]
            u = all_results[uk]
            logger.info(f"  {dim:<5s} {s['rho']:>8.3f} {s['mse']:>8.4f} {u['rho']:>8.3f} {u['mse']:>8.4f}")

    logger.info(f"\nTotal time: {total_time/60:.1f} minutes")

    # Save all results
    results_file = RESULTS_DIR / "camembert_extractors_results.json"
    # Convert numpy to float for JSON
    clean_results = {}
    for k, v in all_results.items():
        clean_results[k] = {kk: float(vv) if hasattr(vv, 'item') else vv for kk, vv in v.items()}
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
