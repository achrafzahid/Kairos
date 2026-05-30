"""
backtest.py — Aggressive HFT-style DeepLOB trading simulation.

Mandate (configured):
  - Up to 10x gross leverage on the $10M book (hard cap, clips Kelly).
  - Kelly-style sizing on model edge (fractional-Kelly multiplier, tunable).
  - Dynamic management: ADD to a position when conviction rises, TRIM when it
    fades, FLIP on a confident opposite signal.
  - Uses ALL THREE horizons: agreement across h10/h20/h50 strengthens the edge.
  - Books cleared every day (no overnight risk). Fees = 0 (per request).

Leak-free: decision at window i uses only X[i] (past 100 snapshots). Execution
uses mid[keep[i]+delay] — a price at or after the decision, never the future
mid the label was derived from.

Pluggable for your future modules:
  - Sizer.size(...)        -> shares for a *target* position given edge/equity
  - RiskManager.vet(...)   -> clip/adjust per-trade; book-level leverage cap is
                              applied by the engine on top of whatever risk returns.
Swap the two instances in main(). Defaults implement the aggressive mandate.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

from deeplob import DeepLOB

HORIZONS = [10, 20, 50]
N_TIMESTEPS = 100

CONFIG = {
    "base_dir": "../tensors",
    "tickers": ["AAPL", "SPY", "NVDA"],
    "checkpoint": "best_deeplob_model.pth",
    "initial_capital": 10_000_000.0,

    "test_days": 10,
    "test_dates": None,            # e.g. ["2024-09-30", ...] to pin specific days

    # ---- Aggression / sizing ----
    "max_gross_leverage": 10.0,    # hard cap: total notional <= 10x equity
    "kelly_multiplier": 1.0,       # 1.0 = full Kelly on softmax edge; <1 safer
    "max_position_leverage": 6.0,  # per-name notional cap (x equity)
    "min_edge_to_trade": 0.05,     # Kelly edge below this -> no/*close* position
    "add_threshold": 0.10,         # raise target by >=10% notional -> ADD
    "trim_threshold": 0.10,        # drop target by >=10% -> TRIM
    "entry_delay": 1,              # execute this many snapshots after the signal

    # ---- Horizon fusion ----
    "horizon_weights": [0.2, 0.3, 0.5],  # weight h10/h20/h50 when fusing edge

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose_trades": False,
}


# ============================================================================
# SIZING (pluggable) — default = Kelly on fused multi-horizon edge
# ============================================================================
class Sizer:
    def target_notional(self, equity, edge, side, ctx):
        """Return desired SIGNED notional ($) for the position. +long / -short."""
        raise NotImplementedError


class KellySizer(Sizer):
    """Fractional-Kelly on edge. edge in [-1,1] (already signed by direction).
    notional = kelly_mult * |edge| * equity * max_position_leverage, signed."""
    def __init__(self, kelly_mult, max_pos_lev):
        self.k = kelly_mult
        self.max_pos_lev = max_pos_lev

    def target_notional(self, equity, edge, side, ctx):
        frac = self.k * abs(edge)                 # Kelly fraction of the per-name budget
        frac = min(frac, 1.0)
        notional = frac * equity * self.max_pos_lev
        return side * notional


# ============================================================================
# RISK (pluggable) — default = pass-through (engine still enforces gross cap)
# ============================================================================
class RiskManager:
    def vet(self, equity, ticker, target_notional, price, open_positions, ctx):
        """Return possibly-adjusted target notional for this name."""
        raise NotImplementedError


class PassThroughRiskManager(RiskManager):
    def vet(self, equity, ticker, target_notional, price, open_positions, ctx):
        return target_notional


# ============================================================================
# PORTFOLIO
# ============================================================================
class Position:
    __slots__ = ("side", "entry_vwap", "shares")
    def __init__(self, side, entry_vwap, shares):
        self.side = side
        self.entry_vwap = entry_vwap
        self.shares = shares

    def notional(self, price):
        return self.side * self.shares * price


class Portfolio:
    def __init__(self, initial_capital):
        self.equity = float(initial_capital)
        self.positions = {}      # ticker -> Position
        self.trades = []

    def gross_notional(self, prices):
        return sum(p.shares * prices.get(t, p.entry_vwap)
                   for t, p in self.positions.items())

    def _record(self, day, ticker, action, side, shares, price, pnl, conf, edge):
        self.trades.append({
            "day": day, "ticker": ticker, "action": action,
            "side": "long" if side > 0 else "short",
            "shares": shares, "price": price, "pnl": pnl,
            "conf": conf, "edge": edge,
        })
        if CONFIG["verbose_trades"]:
            print(f"      [{day} {ticker}] {action:5s} {('L' if side>0 else 'S')} "
                  f"{shares:8.0f} @ {price:.4f}  pnl={pnl:+.2f}  edge={edge:+.2f}")

    def set_target_shares(self, day, ticker, side, target_shares, price, conf, edge):
        """Move toward a signed target by adding/trimming/flipping. Returns realized pnl."""
        cur = self.positions.get(ticker)
        realized = 0.0

        if cur is None:
            if target_shares > 0:
                self.positions[ticker] = Position(side, price, target_shares)
                self._record(day, ticker, "OPEN", side, target_shares, price, 0.0, conf, edge)
            return realized

        if side != cur.side:
            # Flip: close all, then open the other way.
            realized += self._close(day, ticker, price, conf, edge, "FLIP")
            if target_shares > 0:
                self.positions[ticker] = Position(side, price, target_shares)
                self._record(day, ticker, "OPEN", side, target_shares, price, 0.0, conf, edge)
            return realized

        # Same side: adjust toward target.
        delta = target_shares - cur.shares
        if delta > 0:        # ADD
            new_total = cur.shares + delta
            cur.entry_vwap = (cur.entry_vwap * cur.shares + price * delta) / new_total
            cur.shares = new_total
            self._record(day, ticker, "ADD", side, delta, price, 0.0, conf, edge)
        elif delta < 0:      # TRIM (realize pnl on the reduced shares)
            qty = -delta
            pnl = cur.side * (price - cur.entry_vwap) * qty
            self.equity += pnl
            realized += pnl
            cur.shares -= qty
            self._record(day, ticker, "TRIM", side, qty, price, pnl, conf, edge)
            if cur.shares <= 1e-9:
                self.positions.pop(ticker, None)
        return realized

    def _close(self, day, ticker, price, conf, edge, reason):
        pos = self.positions.pop(ticker)
        pnl = pos.side * (price - pos.entry_vwap) * pos.shares
        self.equity += pnl
        self._record(day, ticker, reason, pos.side, pos.shares, price, pnl, conf, edge)
        return pnl

    def close(self, day, ticker, price, reason="EOD"):
        if ticker in self.positions:
            return self._close(day, ticker, price, 0.0, 0.0, reason)
        return 0.0


# ============================================================================
# ALIGNMENT  (reproduce ETL window<->signal validity)
# ============================================================================
def _safe_nanmean(arr):
    finite = arr[np.isfinite(arr)]
    return finite.mean() if finite.size > 0 else np.nan


def _reconstruct_keep(signal, n_windows):
    n = len(signal)
    anchor_idx = np.arange(N_TIMESTEPS - 1, n)
    valid = np.ones(len(anchor_idx), dtype=bool)
    for k in HORIZONS:
        for r, t in enumerate(anchor_idx):
            if not valid[r]:
                continue
            if t - k < 0 or t + k >= n:
                valid[r] = False
                continue
            mm = _safe_nanmean(signal[t - k + 1: t + 1])
            mp = _safe_nanmean(signal[t + 1: t + k + 1])
            if not (np.isfinite(mm) and mm != 0 and np.isfinite(mp)):
                valid[r] = False
    keep = anchor_idx[valid]
    if len(keep) != n_windows:
        print(f"    WARN: keep ({len(keep)}) != X rows ({n_windows}); "
              f"using first {n_windows} anchors.")
        keep = anchor_idx[:n_windows]
    return keep


# ============================================================================
# PREDICTION — fuse all 3 horizons into one signed edge in [-1, 1]
# ============================================================================
@torch.no_grad()
def predict_day(model, X, device, batch=512):
    """Return per-window: edge[N] in [-1,1] (signed), conf[N], side[N]."""
    w = np.array(CONFIG["horizon_weights"], dtype=np.float64)
    w = w / w.sum()
    edges, confs, sides = [], [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        probs = F.softmax(model(xb), dim=2).cpu().numpy()   # (b,3,3): [flat,up,down]
        # directional edge per horizon = P(up) - P(down)
        e_h = probs[:, :, 1] - probs[:, :, 2]               # (b,3)
        fused = e_h @ w                                     # (b,) weighted edge
        conf = np.abs(fused)
        side = np.sign(fused)
        edges.append(fused); confs.append(conf); sides.append(side)
    return (np.concatenate(edges), np.concatenate(confs),
            np.concatenate(sides).astype(int))


# ============================================================================
# SIMULATE ONE TICKER-DAY
# ============================================================================
def simulate_ticker_day(model, pf, paths, ticker, day, sizer, risk, device,
                        prices_now):
    xp, mp, sp = paths
    X = torch.load(xp, weights_only=True)
    mid = torch.load(mp, weights_only=True).numpy()
    signal = torch.load(sp, weights_only=True).numpy()

    keep = _reconstruct_keep(signal, len(X))
    mid_exec = mid[keep]
    edge, conf, side = predict_day(model, X, device)

    n = len(edge)
    delay = CONFIG["entry_delay"]
    min_edge = CONFIG["min_edge_to_trade"]
    add_thr = CONFIG["add_threshold"]
    trim_thr = CONFIG["trim_threshold"]

    for i in range(n):
        ei = i + delay
        if ei >= n:
            break
        price = float(mid_exec[ei])
        if not np.isfinite(price) or price <= 0:
            continue
        prices_now[ticker] = price          # keep latest price for gross calc

        e = float(edge[i]); s = int(side[i])

        # No tradable edge -> exit any position (don't sit in noise).
        if abs(e) < min_edge or s == 0:
            if ticker in pf.positions:
                pf.close(day, ticker, price, reason="EXIT")
            continue

        # Desired target notional from Kelly sizer (signed), then risk vet.
        tgt_notional = sizer.target_notional(pf.equity, e, s, ctx={"ticker": ticker})
        tgt_notional = risk.vet(pf.equity, ticker, tgt_notional, price,
                                pf.positions, ctx={"ticker": ticker})

        # ---- Book-level gross leverage cap (engine-enforced backstop) ----
        gross_cap = CONFIG["max_gross_leverage"] * pf.equity
        # gross excluding this name's current position
        gross_others = sum(p.shares * prices_now.get(t, p.entry_vwap)
                           for t, p in pf.positions.items() if t != ticker)
        room = max(0.0, gross_cap - gross_others)
        if abs(tgt_notional) > room:
            tgt_notional = np.sign(tgt_notional) * room

        target_shares = abs(tgt_notional) / price if price > 0 else 0.0
        target_shares = float(int(target_shares))

        cur = pf.positions.get(ticker)
        if cur is None:
            if target_shares > 0:
                pf.set_target_shares(day, ticker, s, target_shares, price, conf[i], e)
        else:
            # ADD / TRIM / FLIP decided inside set_target_shares, but gate
            # add/trim by thresholds to avoid churning on micro-changes.
            if s != cur.side:
                pf.set_target_shares(day, ticker, s, target_shares, price, conf[i], e)
            else:
                cur_notional = cur.shares * price
                want_notional = target_shares * price
                rel = (want_notional - cur_notional) / max(cur_notional, 1.0)
                if rel >= add_thr or rel <= -trim_thr:
                    pf.set_target_shares(day, ticker, s, target_shares, price, conf[i], e)
                # else hold

    # Clear the book for this name at the last executable price.
    if ticker in pf.positions:
        pf.close(day, ticker, float(mid_exec[-1]), reason="EOD")
    prices_now.pop(ticker, None)


# ============================================================================
# DRIVER
# ============================================================================
def discover_test_dates(base_dir, tickers, test_days, override=None):
    if override:
        return list(override)
    files = sorted(glob.glob(os.path.join(base_dir, tickers[0], "*_X.pt")))
    dates = [os.path.basename(f).split('_')[1] for f in files]
    return dates[-test_days:] if test_days <= len(dates) else dates


def run_backtest(model, sizer, risk):
    device = CONFIG["device"]
    base, tickers = CONFIG["base_dir"], CONFIG["tickers"]
    dates = discover_test_dates(base, tickers, CONFIG["test_days"], CONFIG["test_dates"])
    if not dates:
        print(f"No test dates under {base}/{tickers[0]}."); return None

    pf = Portfolio(CONFIG["initial_capital"])
    print(f"\n{'='*78}")
    print(f"AGGRESSIVE BACKTEST | ${pf.equity:,.0f} | gross<= {CONFIG['max_gross_leverage']}x"
          f" | Kelly x{CONFIG['kelly_multiplier']} | fees=0")
    print(f"{'='*78}\nTest days: {dates}\n")

    daily, peak = [], pf.equity
    max_dd = 0.0
    for day in dates:
        eq_before = pf.equity
        tr_before = len(pf.trades)
        prices_now = {}
        for tic in tickers:
            xp = os.path.join(base, tic, f"{tic}_{day}_X.pt")
            mp = os.path.join(base, tic, f"{tic}_{day}_mid.pt")
            sp = os.path.join(base, tic, f"{tic}_{day}_signal.pt")
            if not (os.path.exists(xp) and os.path.exists(mp) and os.path.exists(sp)):
                continue
            simulate_ticker_day(model, pf, (xp, mp, sp), tic, day,
                                sizer, risk, device, prices_now)

        day_pnl = pf.equity - eq_before
        ntr = len(pf.trades) - tr_before
        ret = 100.0 * day_pnl / eq_before if eq_before else 0.0
        peak = max(peak, pf.equity)
        max_dd = min(max_dd, 100.0 * (pf.equity - peak) / peak)
        daily.append((day, day_pnl, pf.equity, ntr, ret))
        print(f"{day} | P&L {day_pnl:+13,.2f} | equity ${pf.equity:15,.2f} "
              f"| {ntr:5d} fills | {ret:+.3f}%")

    total = pf.equity - CONFIG["initial_capital"]
    tret = 100.0 * total / CONFIG["initial_capital"]
    closing = [t for t in pf.trades if t["pnl"] != 0.0]
    wins = [t for t in closing if t["pnl"] > 0]
    wr = 100.0 * len(wins) / len(closing) if closing else 0.0
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in closing if t["pnl"] < 0)
    pf_ratio = (gp / gl) if gl > 0 else float('inf')
    print(f"\n{'-'*78}")
    print(f"TOTAL P&L {total:+,.2f} | final ${pf.equity:,.2f} | return {tret:+.3f}% "
          f"| max DD {max_dd:.2f}%")
    print(f"Fills {len(pf.trades)} | realizing trades {len(closing)} | win {wr:.1f}% "
          f"| profit factor {pf_ratio:.2f}")
    print(f"{'-'*78}\n")
    return {"daily": daily, "trades": pf.trades, "final_equity": pf.equity,
            "return_pct": tret, "max_drawdown_pct": max_dd}


def main():
    device = CONFIG["device"]
    model = DeepLOB(num_horizons=len(HORIZONS)).to(device)
    ckpt = CONFIG["checkpoint"]
    if not os.path.exists(ckpt):
        print(f"Checkpoint '{ckpt}' not found. Train (main.py) first."); return
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {ckpt} on {device}")

    # ---- Swap for your modules later; defaults = aggressive mandate ----
    sizer = KellySizer(kelly_mult=CONFIG["kelly_multiplier"],
                       max_pos_lev=CONFIG["max_position_leverage"])
    risk = PassThroughRiskManager()

    run_backtest(model, sizer, risk)


if __name__ == "__main__":
    main()