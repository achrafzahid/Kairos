import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score
import warnings

from deeplob import DeepLOB

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    # Must point to where load.py wrote the tensors (../tensors/<TICKER>/...).
    # Change this if you run main.py from a different working directory.
    "base_dir": "../tensors",
    "tickers": ["AAPL", "SPY", "NVDA"],
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 0.001,
    "horizons": [10, 20, 50],
    "train_days": 20,
    "val_days": 5,
    "test_days": 5,
}

# ==========================================
# 2. SPLIT RESOLUTION (robust to having fewer days than configured)
# ==========================================
def resolve_splits(n_dates, cfg):
    """Return ((tr_start,tr_len),(va_start,va_len),(te_start,te_len)).
    Uses the configured day counts when enough dates exist; otherwise falls
    back to a 70/15/15 ratio so validation/test are never silently empty."""
    tr, va, te = cfg["train_days"], cfg["val_days"], cfg["test_days"]
    if n_dates >= tr + va + te:
        return (0, tr), (tr, va), (tr + va, te)

    print(f"NOTE: only {n_dates} dates available; configured "
          f"{tr}/{va}/{te} doesn't fit. Falling back to a 70/15/15 split.")
    if n_dates <= 1:
        return (0, n_dates), (0, 0), (0, 0)
    va = max(1, round(n_dates * 0.15))
    te = max(1, round(n_dates * 0.15)) if n_dates >= 4 else 0
    tr = n_dates - va - te
    if tr < 1:
        tr, va, te = n_dates - 1, 1, 0
    return (0, tr), (tr, va), (tr + va, te)

# ==========================================
# 3. DATA & EDGE-CASE ENGINE
# ==========================================
def get_daily_dataloader(x_path, y_path, batch_size, shuffle=True):
    """Loads exactly ONE file pair safely into RAM."""
    try:
        X = torch.load(x_path, weights_only=True)
        Y = torch.load(y_path, weights_only=True)
        if X.shape[0] != Y.shape[0]:
            print(f"SKIP: X/Y length mismatch in {os.path.basename(x_path)} "
                  f"({X.shape[0]} vs {Y.shape[0]})")
            return None
        dataset = TensorDataset(X, Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    except Exception as e:
        print(f"Error loading {x_path}: {e}")
        return None

def compute_failsafe_weights(Y_tensor, num_classes=3):
    """Class weights; won't crash if a class is missing in this day."""
    y_numpy = Y_tensor.cpu().numpy().flatten()
    unique_classes, class_counts = np.unique(y_numpy, return_counts=True)
    total_samples = len(y_numpy)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, count in zip(unique_classes, class_counts):
        if count > 0:
            weights[int(cls)] = total_samples / (num_classes * count)
    return torch.tensor(weights, dtype=torch.float32)

def build_interleaved_schedule(base_dir, tickers, phase_start, phase_length):
    """Pairs tickers on the same date so the model doesn't forget across days."""
    schedule = []
    if phase_length <= 0:
        return schedule

    master_files = sorted(glob.glob(os.path.join(base_dir, tickers[0], "*_X.pt")))
    if not master_files:
        print(f"WARNING: No files found for {tickers[0]} in {base_dir}")
        return schedule

    master_dates = [os.path.basename(f).split('_')[1] for f in master_files]
    target_dates = master_dates[phase_start: phase_start + phase_length]

    for date in target_dates:
        for ticker in tickers:
            x_file = os.path.join(base_dir, ticker, f"{ticker}_{date}_X.pt")
            y_file = os.path.join(base_dir, ticker, f"{ticker}_{date}_Y.pt")
            if os.path.exists(x_file) and os.path.exists(y_file):
                schedule.append((x_file, y_file))
    return schedule

# ==========================================
# 4. EVALUATION ENGINE
# ==========================================
def evaluate_model(model, schedule, device):
    """Strict chronological evaluation over a schedule. Returns mean macro-F1."""
    model.eval()
    all_preds = {0: [], 1: [], 2: []}
    all_trues = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for x_path, y_path in schedule:
            loader = get_daily_dataloader(x_path, y_path, CONFIG["batch_size"], shuffle=False)
            if loader is None:
                continue
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=2).cpu().numpy()
                trues = Y_batch.cpu().numpy()
                for h_idx in range(len(CONFIG["horizons"])):
                    all_preds[h_idx].extend(preds[:, h_idx])
                    all_trues[h_idx].extend(trues[:, h_idx])

    if not all_trues[0]:
        print("--- EVALUATION SKIPPED (no samples in schedule) ---")
        return 0.0

    print("\n--- EVALUATION RESULTS ---")
    total_macro_f1 = 0.0
    for h_idx, horizon in enumerate(CONFIG["horizons"]):
        y_true, y_pred = all_trues[h_idx], all_preds[h_idx]
        f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)
        total_macro_f1 += f1
        # Diagnostic: what is the model actually predicting?
        from collections import Counter
        pred_dist = Counter(y_pred)
        true_dist = Counter(y_true)
        print(f"Horizon {horizon}s | Macro F1: {f1:.4f} | Precision: {precision:.4f} "
              f"| Predicted: {dict(sorted(pred_dist.items()))} "
              f"| Actual: {dict(sorted(true_dist.items()))}")
    return total_macro_f1 / len(CONFIG["horizons"])

# ==========================================
# 5. MAIN TRAINING EXECUTION
# ==========================================
def main_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Multi-Horizon DeepLOB on: {device}")

    # Discover how many dates we actually have, then resolve splits safely.
    master_files = sorted(glob.glob(os.path.join(CONFIG["base_dir"], CONFIG["tickers"][0], "*_X.pt")))
    n_dates = len(master_files)
    if n_dates == 0:
        print(f"CRITICAL ERROR: no *_X.pt files under {CONFIG['base_dir']}/{CONFIG['tickers'][0]}.")
        print("Did you regenerate tensors with the corrected transform/load? "
              "(expected files like AAPL_<date>_X.pt and AAPL_<date>_Y.pt)")
        return

    (tr_s, tr_l), (va_s, va_l), (te_s, te_l) = resolve_splits(n_dates, CONFIG)
    print(f"Found {n_dates} dates -> train {tr_l} / val {va_l} / test {te_l} days")

    train_schedule = build_interleaved_schedule(CONFIG["base_dir"], CONFIG["tickers"], tr_s, tr_l)
    val_schedule = build_interleaved_schedule(CONFIG["base_dir"], CONFIG["tickers"], va_s, va_l)
    test_schedule = build_interleaved_schedule(CONFIG["base_dir"], CONFIG["tickers"], te_s, te_l)

    if not train_schedule:
        print("CRITICAL ERROR: No training data matched. Check paths and formats.")
        return

    print(f"Training on {len(train_schedule)} day/ticker pairs | "
          f"Validating on {len(val_schedule)} | Testing on {len(test_schedule)}")

    model = DeepLOB(num_horizons=len(CONFIG["horizons"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    best_val_f1 = 0.34  # above the ~0.333 single-class-collapse floor

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        batches_processed = 0
        print(f"\n[Epoch {epoch}/{CONFIG['epochs']}] Starting Training...")

        for x_path, y_path in train_schedule:
            day_Y = torch.load(y_path, weights_only=True)
            class_weights = compute_failsafe_weights(day_Y).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            loader = get_daily_dataloader(x_path, y_path, CONFIG["batch_size"])
            if loader is None:
                continue

            for X_batch, Y_batch in loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss_10 = criterion(outputs[:, 0, :], Y_batch[:, 0])
                loss_20 = criterion(outputs[:, 1, :], Y_batch[:, 1])
                loss_50 = criterion(outputs[:, 2, :], Y_batch[:, 2])
                total_loss = loss_10 + loss_20 + loss_50
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += total_loss.item()
                batches_processed += 1

        avg_train_loss = epoch_loss / max(1, batches_processed)
        print(f"Epoch {epoch} Completed | Avg Train Loss: {avg_train_loss:.4f}")

        # Always keep a "last" checkpoint so a model exists in the morning.
        torch.save(model.state_dict(), "last_deeplob_model.pth")

        if val_schedule:
            current_val_f1 = evaluate_model(model, val_schedule, device)
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                torch.save(model.state_dict(), "best_deeplob_model.pth")
                print(">>> NEW BEST MODEL SAVED! <<<")

    # Final test pass on the best checkpoint (falls back to last).
    if test_schedule:
        ckpt = "best_deeplob_model.pth" if os.path.exists("best_deeplob_model.pth") else "last_deeplob_model.pth"
        print(f"\n=== FINAL TEST (loading {ckpt}) ===")
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        evaluate_model(model, test_schedule, device)

    print("\nALL DONE.")

# ==========================================
# 6. DUMMY TEST BLOCK
# ==========================================
def run_dummy_test():
    print("\n" + "=" * 50 + "\nSTARTING DUMMY TEST OVERRIDE\n" + "=" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Hardware: {device}")
    try:
        model = DeepLOB(num_horizons=3, num_classes=3).to(device)
        print("SUCCESS: Model Instantiated.")
    except Exception as e:
        print(f"FAIL: Architecture Error: {e}"); return

    dummy_x = torch.randn(64, 100, 40).to(device)
    dummy_y = torch.randint(0, 3, (64, 3)).to(device)
    print(f"Fake X Input Shape: {dummy_x.shape}")
    print(f"Fake Y Target Shape: {dummy_y.shape}")
    try:
        outputs = model(dummy_x)
        print(f"SUCCESS: Forward Pass Complete. Output Shape: {outputs.shape}")
    except Exception as e:
        print(f"FAIL: Forward Pass Error: {e}"); return
    try:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        total_loss = (criterion(outputs[:, 0, :], dummy_y[:, 0])
                      + criterion(outputs[:, 1, :], dummy_y[:, 1])
                      + criterion(outputs[:, 2, :], dummy_y[:, 2]))
        print(f"SUCCESS: Total Loss Calculated: {total_loss.item():.4f}")
        total_loss.backward(); optimizer.step()
        print("SUCCESS: Backward Pass & Optimizer Step Complete!")
    except Exception as e:
        print(f"FAIL: Backward Pass Error: {e}"); return
    print("=" * 50 + "\nDUMMY TEST PASSED!\n" + "=" * 50 + "\n")

# ==========================================
# EXECUTION TOGGLE
# ==========================================
if __name__ == "__main__":
    # run_dummy_test()
    main_training_pipeline()