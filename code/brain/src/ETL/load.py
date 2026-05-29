import torch
import os
import glob

TICKER = "SPY"
DOWNLOAD_DIR = "../pcap"
PARSED_DIR = "../parsed"
TENSORS_DIR = "../tensors"


def load_tensor(normalized_tensor, ticker, date_str):
    """Persist the dict returned by build_and_save_deeplob_tensors.

    normalized_tensor = {"X", "Y", "mid", "micro"} or None.
    """
    if normalized_tensor is None:
        print(f"[{ticker}] Nothing to save for {date_str} (empty data).")
        return

    save_dir = os.path.join(TENSORS_DIR, ticker)
    os.makedirs(save_dir, exist_ok=True)

    X = torch.as_tensor(normalized_tensor["X"], dtype=torch.float32)
    Y = torch.as_tensor(normalized_tensor["Y"], dtype=torch.long)
    mid = torch.as_tensor(normalized_tensor["mid"], dtype=torch.float32)

    torch.save(X, os.path.join(save_dir, f"{ticker}_{date_str}_X.pt"))
    torch.save(Y, os.path.join(save_dir, f"{ticker}_{date_str}_Y.pt"))
    torch.save(mid, os.path.join(save_dir, f"{ticker}_{date_str}_mid.pt"))

    if "signal" in normalized_tensor:
        sig = torch.as_tensor(normalized_tensor["signal"], dtype=torch.float32)
        torch.save(sig, os.path.join(save_dir, f"{ticker}_{date_str}_signal.pt"))

    print(f"[{ticker}] SUCCESS {date_str}: "
          f"X={tuple(X.shape)} Y={tuple(Y.shape)} mid={tuple(mid.shape)} -> {save_dir}")


def remove_csv(csv_path):
    try:
        os.remove(csv_path)
    except (OSError, TypeError):
        pass  # already gone / called more than once for the same day


def remove_pcap():
    pcap_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pcap*"))
    for file_path in pcap_files:
        try:
            os.remove(file_path)
            print(f"Successfully deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")