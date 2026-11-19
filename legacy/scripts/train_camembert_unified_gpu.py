"""
CamemBERT Risk Scoring for Moroccan Misinformation Claims.
Single regression head: predicts global risk_score (0-1).
Train on 974 LLM-scored claims, evaluate on 26 human-annotated gold claims.

Usage:
    cd ~/Desktop/paper2
    source venv/bin/activate
    python scripts/train_camembert_unified_gpu.py
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "camembert-base"
MAX_LEN = 128
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 15
LR = 2e-5
WARMUP_STEPS = 100

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    logger.warning("CUDA not available! Running on CPU (will be slow)")

USE_AMP = torch.cuda.is_available()

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

logger.info(f"Device: {DEVICE}")
logger.info(f"Using mixed precision: {USE_AMP}")
logger.info(f"Batch size: {BATCH_SIZE}")

# ─── DATASET ─────────────────────────────────────────────────────────────────
class MoroccanClaimsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item.get('claim', '')

        encoded = self.tokenizer(
            claim,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'risk_score': torch.tensor(item.get('risk_score', 0.5), dtype=torch.float32),
        }

# ─── MODEL ───────────────────────────────────────────────────────────────────
class CamemBERTRiskScorer(nn.Module):
    def __init__(self, encoder_dim=768):
        super().__init__()

        # Shared encoder (CamemBERT)
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        # Single regression head: risk_score (0-1)
        self.risk_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask).pooler_output
        risk_score = self.risk_head(h)
        return risk_score

# ─── TRAINING ────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, scaler, scheduler):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        risk_score = batch['risk_score'].to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            pred = model(input_ids, attention_mask)
            loss = nn.MSELoss()(pred, risk_score) / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)

def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            with autocast(enabled=USE_AMP):
                pred = model(input_ids, attention_mask)

            all_preds.extend(pred.squeeze().cpu().numpy().tolist())
            all_true.extend(batch['risk_score'].numpy().tolist())

    # Handle single-element batches
    if not isinstance(all_preds, list):
        all_preds = [all_preds]
    if not isinstance(all_true, list):
        all_true = [all_true]

    rho, p_value = spearmanr(all_preds, all_true)
    mse = mean_squared_error(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)

    return rho, p_value, mse, mae

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading data...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load training data (974 original + 2030 LLM-rescored synthetic = 3004 claims)
    train_data_path = DATA_DIR / "final" / "training_data_llm_extended.json"
    train_dataset = MoroccanClaimsDataset(train_data_path, tokenizer, MAX_LEN)

    # Load test data (26 gold-standard human-annotated claims)
    test_data_path = DATA_DIR / "final" / "gold_test.json"
    test_dataset = MoroccanClaimsDataset(test_data_path, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                             num_workers=4, pin_memory=True)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Model, optimizer, scheduler
    model = CamemBERTRiskScorer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=total_steps)
    scaler = GradScaler(enabled=USE_AMP)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training loop
    best_rho = -1
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler)
        rho, p_value, mse, mae = evaluate(model, test_loader)
        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")
        logger.info(f"  Spearman ρ: {rho:.3f} (p={p_value:.4f})")
        logger.info(f"  MSE: {mse:.4f} | MAE: {mae:.4f}")
        logger.info(f"  Time: {epoch_time:.1f}s")

        if rho > best_rho:
            best_rho = rho
            torch.save(model.state_dict(), OUTPUT_DIR / "camembert_risk_best.pt")
            logger.info(f"  ✓ Saved best model (ρ={best_rho:.3f})")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best Spearman ρ: {best_rho:.3f}")
    logger.info(f"Model saved to: {OUTPUT_DIR / 'camembert_risk_best.pt'}")
    logger.info(f"{'='*60}")
