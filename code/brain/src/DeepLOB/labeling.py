import os
import glob
import torch
import numpy as np

# Put the function we discussed earlier here
def generate_multi_horizon_labels(raw_tensor, horizons=[10, 20, 50], alpha=0.00005):
    """Generates the Y labels for the 3 horizons."""
    best_asks = raw_tensor[:, 99, 0].numpy()
    best_bids = raw_tensor[:, 99, 2].numpy()
    mid_prices = (best_asks + best_bids) / 2.0
    
    num_samples = len(mid_prices)
    labels = np.zeros((num_samples, len(horizons)))
    
    for t in range(num_samples):
        for idx, k in enumerate(horizons):
            if t < k or t + k >= num_samples:
                labels[t, idx] = 0 
                continue
                
            m_past = np.mean(mid_prices[t-k : t])
            m_future = np.mean(mid_prices[t+1 : t+k+1])
            pct_change = (m_future - m_past) / m_past
            
            if pct_change > alpha:
                labels[t, idx] = 1  # UP
            elif pct_change < -alpha:
                labels[t, idx] = 2  # DOWN
            else:
                labels[t, idx] = 0  # FLAT
                
    return torch.tensor(labels, dtype=torch.long)

# ==========================================
# RUN THE CONVERSION LOOP
# ==========================================
TENSORS_DIR = "../../data/AAPL" # Adjust to your folder
alpha_threshold = 0.0000005      # Tune this if your distribution is mostly FLAT

print(f"Scanning {TENSORS_DIR} for raw tensors...")
raw_files = glob.glob(os.path.join(TENSORS_DIR, "*.pt"))

# Filter out files that already have _X or _Y in the name
files_to_process = [f for f in raw_files if not (f.endswith("_Y.pt"))]

for file_path in files_to_process:
    print(f"\nProcessing {file_path}...")
    
    # 1. Load the raw X tensor
    raw_X = torch.load(file_path)
    
    # 2. Generate the Y labels using the raw X
    Y_labels = generate_multi_horizon_labels(raw_X, alpha=alpha_threshold)
    
    # 3. Create the new filenames
    # Replaces 'AAPL_2024-07-05.pt' with 'AAPL_2024-07-05_X.pt'
    base_name = file_path.replace(".pt", "")
    x_filename = f"{base_name}_X.pt"
    y_filename = f"{base_name}_Y.pt"
    
    # 4. Save the X and Y files
    torch.save(raw_X, x_filename)
    torch.save(Y_labels, y_filename)
    print(f"Saved: {x_filename} and {y_filename}")
    
    # 5. Delete the old original file so it doesn't waste SSD space
    os.remove(file_path)

print("\n--- ALL FILES CONVERTED AND READY FOR TRAINING ---")