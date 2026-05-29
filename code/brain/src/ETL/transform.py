import os
import glob
import numpy as np
import pandas as pd
from iex_cppparser import parse_dates  # kept for parity; unused here

# --- Configurations ---
DOWNLOAD_DIR = "../pcap"
PARSED_DIR = "../parsed"
TENSORS_DIR = "../tensors"

HORIZONS = [10, 20, 50]   # prediction horizons in 1-second snapshots (matches deeplob.py)
N_TIMESTEPS = 100         # the 100 most recent LOB states per input
LEVELS = 10
N_FEATURES = 4 * LEVELS   # 40 -> [ask_p, ask_v, bid_p, bid_v] x 10

for d in [DOWNLOAD_DIR, PARSED_DIR, TENSORS_DIR]:
    os.makedirs(d, exist_ok=True)


def return_csv_path(current_date: str):
    search_pattern = os.path.join(PARSED_DIR, f"*{current_date.replace('-', '')}*_prl.csv")
    prl_files = glob.glob(search_pattern)
    if not prl_files:
        print(f"No parsed data found for {current_date}. Skipping.")
        return None
    print(f"csv file is {prl_files[0]}")
    return prl_files[0]


def cleanday(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df = df[df["Record Type"] == "R"]
    df = df[df["Event Flag"] == 1]
    # IMPORTANT: do NOT drop Size == 0. In IEX price-level updates a size of 0
    # means the level was removed; the book builder needs these to pop stale
    # levels. Dropping them corrupts the best bid/ask and therefore the mid.
    df = df.drop(
        columns=["Event Flag", "Record Type", "Packet Capture Time",
                 "Send Time", "Tick Type"],
        errors="ignore",
    )

    AAPL_df = df[df["Symbol"] == "AAPL"].drop(columns=["Symbol"])
    NVDA_df = df[df["Symbol"] == "NVDA"].drop(columns=["Symbol"])
    SPY_df = df[df["Symbol"] == "SPY"].drop(columns=["Symbol"])
    return {"AAPL": AAPL_df, "NVDA": NVDA_df, "SPY": SPY_df}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_second_snapshots(df: pd.DataFrame):
    """Clock-time: one snapshot per elapsed second (forward-fills dead air).
    Use when the feed is dense (>10k events/day). For sparse feeds like IEX,
    prefer _build_event_snapshots."""
    if df is None or len(df) == 0:
        return None, None

    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Timestamp"] = df["Exchange Timestamp"]
    df["Flag"] = df["Buy_Ask Flag"]
    df = df.drop(columns=["Exchange Timestamp", "Buy_Ask Flag"], errors="ignore")
    df = df.sort_values(by="Timestamp")
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="ns", utc=True)
    df["Second"] = df["Datetime"].dt.floor("s")

    bids, asks = {}, {}
    snapshots, mids = [], []
    current_second = None

    def take_snapshot():
        top_bids = sorted(bids.items(), key=lambda kv: kv[0], reverse=True)[:LEVELS]
        top_asks = sorted(asks.items(), key=lambda kv: kv[0])[:LEVELS]
        feats = []
        for i in range(LEVELS):
            pa, va = top_asks[i] if i < len(top_asks) else (0.0, 0.0)
            pb, vb = top_bids[i] if i < len(top_bids) else (0.0, 0.0)
            feats.extend([pa, va, pb, vb])
        best_ask = top_asks[0][0] if top_asks else 0.0
        best_bid = top_bids[0][0] if top_bids else 0.0
        mid = 0.5 * (best_ask + best_bid) if (best_ask > 0 and best_bid > 0) else np.nan
        return feats, mid

    for row in df.itertuples():
        price = float(row.Price)
        volume = float(row.Size)
        is_bid = (row.Flag == 1)

        if current_second is None:
            current_second = row.Second

        while current_second < row.Second:
            feats, mid = take_snapshot()
            snapshots.append(feats)
            mids.append(mid)
            current_second += pd.Timedelta(seconds=1)

        book = bids if is_bid else asks
        if volume == 0:
            book.pop(price, None)
        else:
            book[price] = volume

    if not snapshots:
        return None, None
    return np.asarray(snapshots, dtype=np.float64), np.asarray(mids, dtype=np.float64)


def _build_event_snapshots(df: pd.DataFrame, every_n=50, depth=10):
    """Event-time with sub-sampling.  Returns (snapshots, mids, signal).

    `signal` is a DEEP-BOOK VOLUME-WEIGHTED PRICE:
        signal = (Vbid_total * p_ask_L1 + Vask_total * p_bid_L1)
                 / (Vbid_total + Vask_total)
    where Vbid_total / Vask_total are summed over the top `depth` levels.

    Rationale: on IEX, mega-caps (AAPL, NVDA) have almost no top-of-book
    activity — the mid and the L1 micro-price barely move. But hundreds of
    thousands of events churn at *deeper* levels each day. This signal reacts
    to that deep-level volume imbalance, which the paper identifies as a strong
    predictor of the next price move. It moves thousands of times per day even
    when the mid moves twice, giving the model a learnable target.

    `mids` is kept only for reference / trading-sim use, not for labels.
    """
    if df is None or len(df) == 0:
        return None, None, None

    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Timestamp"] = df["Exchange Timestamp"]
    df["Flag"] = df["Buy_Ask Flag"]
    df = df.drop(columns=["Exchange Timestamp", "Buy_Ask Flag"], errors="ignore")
    df = df.sort_values(by="Timestamp")

    bids, asks = {}, {}
    snapshots, mids, signal = [], [], []
    event_count = 0
    mid_changes = 0
    sig_changes = 0
    last_mid = None
    last_sig = None

    for row in df.itertuples():
        price = float(row.Price)
        volume = float(row.Size)
        is_bid = (row.Flag == 1)

        book = bids if is_bid else asks
        if volume == 0:
            book.pop(price, None)
        else:
            book[price] = volume

        if not bids or not asks:
            continue

        event_count += 1
        if event_count % every_n != 0:
            continue

        top_bids = sorted(bids.items(), key=lambda kv: kv[0], reverse=True)[:LEVELS]
        top_asks = sorted(asks.items(), key=lambda kv: kv[0])[:LEVELS]
        feats = []
        for i in range(LEVELS):
            pa, va = top_asks[i] if i < len(top_asks) else (0.0, 0.0)
            pb, vb = top_bids[i] if i < len(top_bids) else (0.0, 0.0)
            feats.extend([pa, va, pb, vb])

        best_ask_p = top_asks[0][0]
        best_bid_p = top_bids[0][0]
        mid = 0.5 * (best_ask_p + best_bid_p)

        # Deep-book volume-weighted price (uses top `depth` levels)
        Vbid = sum(v for _, v in top_bids[:depth])
        Vask = sum(v for _, v in top_asks[:depth])
        if Vbid + Vask > 0:
            sig = (Vbid * best_ask_p + Vask * best_bid_p) / (Vbid + Vask)
        else:
            sig = mid

        if last_mid is not None and mid != last_mid:
            mid_changes += 1
        if last_sig is not None and abs(sig - last_sig) > 1e-12:
            sig_changes += 1
        last_mid, last_sig = mid, sig

        snapshots.append(feats)
        mids.append(mid)
        signal.append(sig)

    if not snapshots:
        return None, None, None
    print(f"    {event_count} total events, every_n={every_n} -> "
          f"{len(snapshots)} snapshots | "
          f"mid changes: {mid_changes}, weighted-price changes: {sig_changes}")
    return (np.asarray(snapshots, dtype=np.float64),
            np.asarray(mids, dtype=np.float64),
            np.asarray(signal, dtype=np.float64))

    if not snapshots:
        return None, None
    return np.asarray(snapshots, dtype=np.float64), np.asarray(mids, dtype=np.float64)


def _safe_nanmean(arr):
    """nanmean that returns NaN without warnings when the slice is all-NaN."""
    finite = arr[np.isfinite(arr)]
    return finite.mean() if finite.size > 0 else np.nan


def _suggest_alpha(mids, anchor_idx, horizons, target_flat=0.34):
    """Choose alpha so ~target_flat of samples are 'flat' at the longest horizon.
    Returns at least 1e-7 so floating-point noise is never classified as a move."""
    k = max(horizons)
    n = len(mids)
    ls = []
    for t in anchor_idx:
        if t - k < 0 or t + k >= n:
            continue
        m_minus = _safe_nanmean(mids[t - k + 1: t + 1])
        m_plus = _safe_nanmean(mids[t + 1: t + k + 1])
        if np.isfinite(m_minus) and m_minus != 0 and np.isfinite(m_plus):
            ls.append(abs((m_plus - m_minus) / m_minus))
    if not ls:
        return 1e-5
    return max(float(np.quantile(np.asarray(ls), target_flat)), 1e-7)


def _make_labels(mids, anchor_idx, horizons, alpha):
    """Threshold-based: l > alpha → up, l < -alpha → down, else flat."""
    n = len(mids)
    Y = np.zeros((len(anchor_idx), len(horizons)), dtype=np.int64)
    valid = np.ones(len(anchor_idx), dtype=bool)
    for col, k in enumerate(horizons):
        for r, t in enumerate(anchor_idx):
            if t - k < 0 or t + k >= n:
                valid[r] = False
                continue
            m_minus = _safe_nanmean(mids[t - k + 1: t + 1])
            m_plus = _safe_nanmean(mids[t + 1: t + k + 1])
            if not (np.isfinite(m_minus) and m_minus != 0 and np.isfinite(m_plus)):
                valid[r] = False
                continue
            l = (m_plus - m_minus) / m_minus
            if l > alpha:
                Y[r, col] = 1
            elif l < -alpha:
                Y[r, col] = 2
            else:
                Y[r, col] = 0
    return Y, valid


def _make_labels_quantile(mids, anchor_idx, horizons):
    """Rank-based: sort all returns per horizon, bottom third = down,
    top third = up, middle third = flat.  Guarantees ~33/34/33 balance
    even when the mid-price is constant for long stretches."""
    n = len(mids)
    n_anchors = len(anchor_idx)
    # First pass: compute l for every anchor x horizon, track validity.
    L = np.full((n_anchors, len(horizons)), np.nan, dtype=np.float64)
    valid = np.ones(n_anchors, dtype=bool)
    for col, k in enumerate(horizons):
        for r, t in enumerate(anchor_idx):
            if t - k < 0 or t + k >= n:
                valid[r] = False
                continue
            m_minus = _safe_nanmean(mids[t - k + 1: t + 1])
            m_plus = _safe_nanmean(mids[t + 1: t + k + 1])
            if not (np.isfinite(m_minus) and m_minus != 0 and np.isfinite(m_plus)):
                valid[r] = False
                continue
            L[r, col] = (m_plus - m_minus) / m_minus

    # Second pass: rank-based labeling on valid rows only.
    Y = np.zeros((n_anchors, len(horizons)), dtype=np.int64)
    valid_idx = np.where(valid)[0]
    for col in range(len(horizons)):
        vals = L[valid_idx, col]
        finite_mask = np.isfinite(vals)
        finite_pos = valid_idx[finite_mask]
        finite_vals = vals[finite_mask]
        N = finite_vals.size
        if N == 0:
            continue
        # Stable argsort → rank from 0..N-1; bottom third down, top third up
        ranks = np.argsort(np.argsort(finite_vals, kind='stable'), kind='stable')
        third = N // 3
        for i, r in enumerate(finite_pos):
            if ranks[i] < third:
                Y[r, col] = 2    # down  (lowest returns)
            elif ranks[i] >= N - third:
                Y[r, col] = 1    # up    (highest returns)
            else:
                Y[r, col] = 0    # flat
    return Y, valid


def _standardize(snaps):
    """Per-column z-score on the de-duplicated per-second array (zeros = empty
    levels, excluded from the statistics)."""
    norm = snaps.astype(np.float64).copy()
    for c in range(N_FEATURES):
        col = norm[:, c]
        nz = col[col != 0]
        if nz.size == 0:
            continue
        m, s = nz.mean(), nz.std()
        if s == 0:
            s = 1.0
        norm[:, c] = (col - m) / s
    return norm


# ---------------------------------------------------------------------------
# Public: same name/signature as before (plus optional alpha). Returns a dict
# {"X", "Y", "mid"} that load_tensor() persists, or None if there is no data.
# ---------------------------------------------------------------------------
def build_and_save_deeplob_tensors(df: pd.DataFrame, ticker, date_str,
                                    alpha=None, label_method="quantile",
                                    snapshot_mode="event", every_n=50):
    """Build normalised DeepLOB inputs + labels from a single ticker/day.

    Labels are derived from a DEEP-BOOK VOLUME-WEIGHTED PRICE (see
    _build_event_snapshots), not the raw mid.  On IEX the mid barely moves for
    mega-caps, but the deep-book weighted price reacts to volume imbalance
    across all 10 levels — where the activity actually is — giving a learnable
    target.

    label_method:
        "quantile" (default) — rank-based top/bottom 33%; guarantees balance.
        "alpha"              — threshold-based (Eq. 4); pass alpha= or auto.

    snapshot_mode: "event" (default) or "clock".
    every_n (default 50): event-mode sub-sampling; horizon k spans k*every_n events.
    """
    print(f"[{ticker}] Starting processing for {date_str} "
          f"(snapshot={snapshot_mode}, every_n={every_n}, labels={label_method})...")

    if snapshot_mode == "event":
        result = _build_event_snapshots(df, every_n=every_n)
        if result[0] is None:
            print(f"[{ticker}] Warning: not enough data to form labelled windows.")
            return None
        snaps, mids, signal = result
    else:
        snaps, mids = _build_second_snapshots(df)
        signal = mids  # clock-time: no deep signal tracked, fall back to mid

    if snaps is None or snaps.shape[0] <= N_TIMESTEPS + max(HORIZONS):
        print(f"[{ticker}] Warning: not enough data to form labelled windows.")
        return None

    n = snaps.shape[0]
    anchor_idx = np.arange(N_TIMESTEPS - 1, n)

    # Guard: if the label signal is essentially constant, this ticker/day has
    # no learnable price signal. Skip it rather than poison training with an
    # all-flat block (which collapses the model to predicting class 0).
    sig_unique = len(np.unique(np.round(signal[np.isfinite(signal)], 10)))
    if sig_unique < 10:
        print(f"[{ticker}] SKIP {date_str}: weighted price nearly constant "
              f"({sig_unique} unique values) — no learnable signal.")
        return None

    # Labels from the deep-book weighted price, BEFORE normalisation.
    if label_method == "alpha":
        if alpha is None:
            alpha = _suggest_alpha(signal, anchor_idx, HORIZONS)
            print(f"[{ticker}] auto alpha = {alpha:.3e}")
        Y, valid = _make_labels(signal, anchor_idx, HORIZONS, alpha)
    else:
        print(f"[{ticker}] Quantile labeling on deep-book weighted price")
        Y, valid = _make_labels_quantile(signal, anchor_idx, HORIZONS)

    norm = _standardize(snaps)

    keep = anchor_idx[valid]
    if keep.size == 0:
        print(f"[{ticker}] No valid labelled windows.")
        return None

    X = np.stack([norm[t - N_TIMESTEPS + 1: t + 1] for t in keep]).astype(np.float32)
    Yv = Y[valid].astype(np.int64)

    print(f"[{ticker}] Built {X.shape[0]} windows. X={tuple(X.shape)} Y={tuple(Yv.shape)}")
    for c, k in enumerate(HORIZONS):
        u, cnt = np.unique(Yv[:, c], return_counts=True)
        dist = {int(a): int(b) for a, b in zip(u, cnt)}
        print(f"    h={k:<3d} dist (0=flat,1=up,2=down): {dist}")

    return {"X": X, "Y": Yv,
            "mid": mids.astype(np.float32),
            "signal": signal.astype(np.float32)}