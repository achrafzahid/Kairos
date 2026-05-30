"""
backtest.py — DeepLOB trading simulation (paper Section V-D, adapted to IEX).

What it does
------------
- Loads a trained DeepLOB checkpoint and the test-day tensors written by the ETL
  (_X.pt, _Y.pt, _mid.pt, _signal.pt).
- Generates causal predictions: each decision uses only the 100-snapshot window
  ending at that point — never future information.
- Trades the EXECUTABLE MID (_mid.pt), not the signal the labels came from.
- Closes all positions before end of day ("clears the books").
- Reports trades and end-of-day equity for each day.
- Fees = 0 (per request).  Sizing and risk are pluggable (see Sizer / RiskManager).

Leak-free by construction
-------------------------
prediction[i] depends on X[i] = snapshots up to anchor t_i.  We execute at
mid[t_i] (optionally t_i + entry_delay).  The label was built from mid AFTER t_i,
but the model never sees it at decision time, and the sim never uses it to trade.

Integration points for later
-----------------------------
- Sizer.size(...)         -> how many shares to put on  (plug your sizing algo)
- RiskManager.allow(...)  -> vet/clip/reject a proposed trade  (plug your risk code)
Both have trivial defaults so this runs today and your modules drop in later.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

from deeplob import DeepLOB

# ----------------------------------------------------------------------------
# Must match the ETL (transform.py) so window<->mid alignment is reconstructed
# exactly without needing to re-run the ETL or save extra files.
# ----------------------------------------------------------------------------
HORIZONS = [10, 20, 50]
N_TIMESTEPS = 100


# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    "base_dir": "../tensors",
    "tickers": ["AAPL", "SPY", "NVDA"],
    "checkpoint": "best_deeplob_model.pth",
    "initial_capital": 10_000_000.0,

    # Which test days to use: the LAST `test_days` dates found (chronological).
    # Set test_dates explicitly to override (list of 'YYYY-MM-DD').
    "test_days": 10,
    "test_dates": None,

    # Trading rule
    "horizon_index": 2,        # 0=h10, 1=h20, 2=h50. Longer = more robust (paper).
    "conf_threshold": 0.50,    # only act when softmax prob >= this; else hold.
    "entry_delay": 5,          # execute `delay` snapshots after the signal (slippage).
    "close_on_flat": False,    # if True, a confident 'flat' closes the position.
    "stop_loss_pct": 0.0,      # 0 = disabled; e.g. 0.02 closes on 2% adverse move from entry

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose_trades": False,   # print every trade (chatty); False prints per-day only.
}


# ============================================================================
# PLUGGABLE SIZING  — replace with your sizing algo later
# ============================================================================
class Sizer:
    """Return the number of shares to trade for a new position."""
    def size(self, equity, price, side, confidence, ctx):
        raise NotImplementedError


class FixedFractionSizer(Sizer):
    """Allocate a fixed fraction of current equity as notional per position.
    fraction=0.10 on $10M at $227 -> ~4405 shares. Simple, compounding."""
    def __init__(self, fraction=0.10):
        self.fraction = fraction

    def size(self, equity, price, side, confidence, ctx):
        if price <= 0:
            return 0.0
        notional = self.fraction * equity
        return float(int(notional / price))   # whole shares


class FixedSharesSizer(Sizer):
    """Always trade the same share count (paper used mu=1)."""
    def __init__(self, shares=1.0):
        self.shares = shares

    def size(self, equity, price, side, confidence, ctx):
        return float(self.shares)


class ConfidenceWeightedSizer(Sizer):
    """Scale size linearly from 0 at conf_threshold to fraction at confidence=1.0."""
    def __init__(self, fraction=0.10, conf_threshold=0.50):
        self.fraction = fraction
        self.conf_threshold = conf_threshold

    def size(self, equity, price, side, confidence, ctx):
        if price <= 0 or confidence <= self.conf_threshold:
            return 0.0
        ramp = (confidence - self.conf_threshold) / (1.0 - self.conf_threshold)
        notional = self.fraction * equity * ramp
        return float(int(notional / price))


class VolatilityScaledSizer(Sizer):
    """Inverse-scale by trailing mid-price volatility.  high vol -> smaller size."""
    def __init__(self, base_fraction=0.10, lookback=50, vol_scale=100.0):
        self.base_fraction = base_fraction
        self.lookback = lookback
        self.vol_scale = vol_scale

    def size(self, equity, price, side, confidence, ctx):
        if price <= 0:
            return 0.0
        recent = ctx.get("recent_mids", [])
        if len(recent) < self.lookback:
            return float(int(self.base_fraction * equity / price))
        rets = np.diff(recent[-self.lookback:]) / recent[-self.lookback:-1]
        vol = np.std(rets)
        scale = 1.0 / (1.0 + self.vol_scale * vol)
        notional = self.base_fraction * equity * scale
        return float(int(notional / price))


class KellySizer(Sizer):
    """Fractional Kelly sizing based on empirical win rate and avg win/loss ratio.
    Needs at least `min_trades` in ctx['trade_history'] before sizing trades."""
    def __init__(self, max_fraction=0.20, min_trades=10):
        self.max_fraction = max_fraction
        self.min_trades = min_trades

    def size(self, equity, price, side, confidence, ctx):
        if price <= 0:
            return 0.0
        history = ctx.get("trade_history", [])
        if len(history) < self.min_trades:
            return 0.0
        wins = [t for t in history if t["pnl"] > 0]
        losses = [t for t in history if t["pnl"] <= 0]
        win_rate = len(wins) / len(history)
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1.0
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        kelly = win_rate - (1.0 - win_rate) * (avg_loss / avg_win)
        kelly = max(0.0, min(kelly, self.max_fraction))
        notional = kelly * equity
        return float(int(notional / price))


# ============================================================================
# PLUGGABLE RISK MANAGEMENT — replace with your risk code later
# ============================================================================
class RiskManager:
    """Vet a proposed trade. Return adjusted share count (0.0 = reject)."""
    def allow(self, equity, price, side, shares, open_positions, ctx):
        raise NotImplementedError


class PassThroughRiskManager(RiskManager):
    """No constraints — accept every trade as proposed."""
    def allow(self, equity, price, side, shares, open_positions, ctx):
        return shares


class StopLossRiskManager(RiskManager):
    """Block re-entry in a direction that just got stopped out."""
    def __init__(self, stop_loss_pct=0.02):
        self.stop_loss_pct = stop_loss_pct

    def allow(self, equity, price, side, shares, open_positions, ctx):
        ticker = ctx.get("ticker")
        pos = open_positions.get(ticker)
        if pos:
            pnl_pct = pos.side * (price - pos.entry) / pos.entry
            if pnl_pct <= -self.stop_loss_pct:
                return 0.0
        return shares


class MaxExposureRiskManager(RiskManager):
    """Cap total notional across all tickers as a fraction of equity."""
    def __init__(self, max_exposure_pct=0.30):
        self.max_exposure = max_exposure_pct

    def allow(self, equity, price, side, shares, open_positions, ctx):
        current = sum(p.shares * p.entry for p in open_positions.values())
        new_notional = shares * price
        total = current + new_notional
        if total > self.max_exposure * equity:
            remaining = self.max_exposure * equity - current
            if remaining <= 0:
                return 0.0
            shares = float(int(remaining / price))
        return shares


class DrawdownLimitRiskManager(RiskManager):
    """Stop all trading when equity drops X% below trailing peak."""
    def __init__(self, max_drawdown_pct=0.10):
        self.max_drawdown = max_drawdown_pct
        self.peak_equity = None

    def allow(self, equity, price, side, shares, open_positions, ctx):
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity
        if dd >= self.max_drawdown:
            return 0.0
        return shares


class VixFilterRiskManager(RiskManager):
    """Skip trading on days with high trailing volatility or daily loss breach."""
    def __init__(self, daily_loss_limit=0.03, vol_threshold=0.005, vol_lookback=20):
        self.daily_loss_limit = daily_loss_limit
        self.vol_threshold = vol_threshold
        self.vol_lookback = vol_lookback
        self._day_start = {}
        self._current_day = None

    def allow(self, equity, price, side, shares, open_positions, ctx):
        day = ctx.get("day")
        if day != self._current_day:
            self._day_start[day] = equity
            self._current_day = day
        day_return = (equity - self._day_start.get(day, equity)) / max(self._day_start.get(day, equity), 1.0)
        if day_return < -self.daily_loss_limit:
            return 0.0

        recent = ctx.get("recent_mids", [])
        if len(recent) > self.vol_lookback:
            rets = np.diff(recent[-self.vol_lookback:]) / recent[-self.vol_lookback-1:-1]
            if np.std(rets) > self.vol_threshold:
                return 0.0
        return shares


# ============================================================================
# PORTFOLIO  (P&L-only model: fees=0, no margin/borrow yet — risk module later)
# ============================================================================
class Position:
    __slots__ = ("side", "entry", "shares")
    def __init__(self, side, entry, shares):
        self.side = side          # +1 long, -1 short
        self.entry = entry        # entry price
        self.shares = shares


class Portfolio:
    def __init__(self, initial_capital):
        self.equity = float(initial_capital)
        self.positions = {}       # ticker -> Position
        self.trades = []          # list of dicts

    def is_flat(self, ticker):
        return ticker not in self.positions

    def open(self, ticker, side, price, shares, t_idx, day, conf):
        self.positions[ticker] = Position(side, price, shares)
        if CONFIG["verbose_trades"]:
            s = "LONG" if side > 0 else "SHORT"
            print(f"      [{day} {ticker}] OPEN  {s:5s} {shares:.0f} @ {price:.4f} "
                  f"(conf {conf:.2f})")

    def close(self, ticker, price, t_idx, day, reason):
        pos = self.positions.pop(ticker)
        pnl = pos.side * (price - pos.entry) * pos.shares   # fees = 0
        self.equity += pnl
        self.trades.append({
            "day": day, "ticker": ticker,
            "side": "long" if pos.side > 0 else "short",
            "entry": pos.entry, "exit": price, "shares": pos.shares,
            "pnl": pnl, "reason": reason,
        })
        if CONFIG["verbose_trades"]:
            print(f"      [{day} {ticker}] CLOSE {'long' if pos.side>0 else 'short':5s} "
                  f"@ {price:.4f}  pnl={pnl:+.2f}  ({reason})")
        return pnl


# ============================================================================
# ALIGNMENT — reproduce the exact window<->mid mapping the ETL used
# ============================================================================
def _safe_nanmean(arr):
    finite = arr[np.isfinite(arr)]
    return finite.mean() if finite.size > 0 else np.nan


def _reconstruct_keep(signal, n_windows):
    """Recompute which snapshot index each X-row corresponds to, using the same
    validity rule the ETL applied (quantile labeling drops rows lacking enough
    future or with non-finite returns). Returns an int array `keep` of length
    n_windows so mid[keep[i]] is the executable price for window i."""
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
            m_minus = _safe_nanmean(signal[t - k + 1: t + 1])
            m_plus = _safe_nanmean(signal[t + 1: t + k + 1])
            if not (np.isfinite(m_minus) and m_minus != 0 and np.isfinite(m_plus)):
                valid[r] = False
    keep = anchor_idx[valid]
    if len(keep) != n_windows:
        # Fall back: if the rule drifted, just take the first n_windows anchors.
        # (Alignment is approximate then — regenerate ETL saving an explicit
        # index if you hit this.)
        print(f"    WARN: reconstructed keep ({len(keep)}) != X rows "
              f"({n_windows}); using first {n_windows} anchors.")
        keep = anchor_idx[:n_windows]
    return keep


# ============================================================================
# PREDICTION
# ============================================================================
@torch.no_grad()
def predict_day(model, X, device, batch=512):
    """Return (preds[N], confs[N]) for the configured horizon."""
    h = CONFIG["horizon_index"]
    preds, confs = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        out = model(xb)                       # (b, 3, 3)
        probs = F.softmax(out, dim=2)         # over classes
        p = probs[:, h, :]                    # (b, 3)
        conf, pred = torch.max(p, dim=1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


# ============================================================================
# CORE: simulate one ticker for one day
# ============================================================================
def simulate_ticker_day(model, pf, x_path, mid_path, signal_path,
                        ticker, day, sizer, risk, device):
    X = torch.load(x_path, weights_only=True)
    mid = torch.load(mid_path, weights_only=True).numpy()
    signal = torch.load(signal_path, weights_only=True).numpy()

    keep = _reconstruct_keep(signal, len(X))
    mid_exec = mid[keep]                       # executable price per window
    preds, confs = predict_day(model, X, device)

    n = len(preds)
    delay = CONFIG["entry_delay"]
    thr = CONFIG["conf_threshold"]
    day_start_equity = pf.equity
    n_trades_before = len(pf.trades)

    for i in range(n):
        # Execute at i+delay if available, else skip acting this step.
        exec_i = i + delay
        if exec_i >= n:
            break
        price = float(mid_exec[exec_i])
        if not np.isfinite(price) or price <= 0:
            continue

        # Stop-loss: close position if adverse move exceeds threshold.
        #  if CONFIG["stop_loss_pct"] > 0:
        #    sl_cur = pf.positions.get(ticker)
        #    if sl_cur is not None:
        #        sl_pnl = sl_cur.side * (price - sl_cur.entry) / sl_cur.entry
        #        if sl_pnl <= -CONFIG["stop_loss_pct"]:
        #            pf.close(ticker, price, exec_i, day, "stop-loss")

        pred = int(preds[i])
        conf = float(confs[i])
        confident = conf >= thr

        # Map class -> desired side. 1=up->long, 2=down->short, 0=flat.
        if confident and pred == 1:
            desired = +1
        elif confident and pred == 2:
            desired = -1
        else:
            desired = 0    # flat or low confidence

        cur = pf.positions.get(ticker)

        if desired == 0:
            if CONFIG["close_on_flat"] and cur is not None and confident and pred == 0:
                pf.close(ticker, price, exec_i, day, "flat-signal")
            continue

        # Have a confident directional signal.
        ctx = {
            "ticker": ticker,
            "recent_mids": mid_exec[max(0, i - 100): i + 1].tolist(),
            "trade_history": pf.trades[-200:],
            "day": day,
        }
        if cur is None:
            shares = sizer.size(pf.equity, price, desired, conf, ctx=ctx)
            shares = risk.allow(pf.equity, price, desired, shares,
                                pf.positions, ctx=ctx)
            if shares and shares > 0:
                pf.open(ticker, desired, price, shares, exec_i, day, conf)
        elif cur.side != desired:
            # Flip: close current, open opposite.
            pf.close(ticker, price, exec_i, day, "flip")
            shares = sizer.size(pf.equity, price, desired, conf, ctx=ctx)
            shares = risk.allow(pf.equity, price, desired, shares,
                                pf.positions, ctx=ctx)
            if shares and shares > 0:
                pf.open(ticker, desired, price, shares, exec_i, day, conf)
        # else: same side already on -> hold.

    # ---- Clear the books: force-close at the last executable price of the day ----
    if not pf.is_flat(ticker):
        last_price = float(mid_exec[-1])
        pf.close(ticker, last_price, n - 1, day, "EOD")

    day_pnl = pf.equity - day_start_equity
    n_trades = len(pf.trades) - n_trades_before
    return day_pnl, n_trades


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
    base = CONFIG["base_dir"]
    tickers = CONFIG["tickers"]

    dates = discover_test_dates(base, tickers, CONFIG["test_days"], CONFIG["test_dates"])
    if not dates:
        print(f"No test dates found under {base}/{tickers[0]}.")
        return None

    pf = Portfolio(CONFIG["initial_capital"])
    print(f"\n{'='*72}\nBACKTEST  |  capital ${pf.equity:,.0f}  |  "
          f"horizon h{HORIZONS[CONFIG['horizon_index']]}  |  "
          f"conf>={CONFIG['conf_threshold']}  |  fees=0\n{'='*72}")
    print(f"Test days: {dates}\n")

    daily = []
    for day in dates:
        eq_before = pf.equity
        trades_before = len(pf.trades)
        per_ticker = {}
        for tic in tickers:
            xp = os.path.join(base, tic, f"{tic}_{day}_X.pt")
            mp = os.path.join(base, tic, f"{tic}_{day}_mid.pt")
            sp = os.path.join(base, tic, f"{tic}_{day}_signal.pt")
            if not (os.path.exists(xp) and os.path.exists(mp) and os.path.exists(sp)):
                continue
            pnl, ntr = simulate_ticker_day(model, pf, xp, mp, sp,
                                           tic, day, sizer, risk, device)
            per_ticker[tic] = (pnl, ntr)

        day_pnl = pf.equity - eq_before
        day_trades = len(pf.trades) - trades_before
        ret_pct = 100.0 * day_pnl / eq_before if eq_before else 0.0
        daily.append((day, day_pnl, pf.equity, day_trades, ret_pct))

        bt = "  ".join(f"{t}:{v[0]:+,.0f}({v[1]})" for t, v in per_ticker.items())
        print(f"{day} | P&L {day_pnl:+12,.2f} | equity ${pf.equity:14,.2f} | "
              f"{day_trades:4d} trades | {ret_pct:+.3f}%   {bt}")

    # ---- Summary ----
    total_pnl = pf.equity - CONFIG["initial_capital"]
    total_ret = 100.0 * total_pnl / CONFIG["initial_capital"]
    wins = [t for t in pf.trades if t["pnl"] > 0]
    win_rate = 100.0 * len(wins) / len(pf.trades) if pf.trades else 0.0
    print(f"\n{'-'*72}")
    print(f"TOTAL  P&L {total_pnl:+,.2f}  |  final equity ${pf.equity:,.2f}  |  "
          f"return {total_ret:+.3f}%")
    print(f"Trades {len(pf.trades)}  |  win rate {win_rate:.1f}%  |  "
          f"avg P&L/trade {('%+.2f' % (total_pnl/len(pf.trades))) if pf.trades else 'n/a'}")
    print(f"{'-'*72}\n")
    return {"daily": daily, "trades": pf.trades, "final_equity": pf.equity}


def main():
    device = CONFIG["device"]
    model = DeepLOB(num_horizons=len(HORIZONS)).to(device)
    ckpt = CONFIG["checkpoint"]
    if not os.path.exists(ckpt):
        print(f"Checkpoint '{ckpt}' not found. Train first (main.py) or set CONFIG['checkpoint'].")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {ckpt} on {device}")

    # ---- Pick one sizer and one risk manager ----
    # sizer = FixedFractionSizer(fraction=0.10)
    # sizer = ConfidenceWeightedSizer(fraction=0.10, conf_threshold=0.50)
    # sizer = VolatilityScaledSizer(base_fraction=0.10, lookback=50, vol_scale=100.0)
    sizer = KellySizer(max_fraction=0.20, min_trades=10)
    # sizer = FixedFractionSizer(fraction=0.10)
    #
    # risk = PassThroughRiskManager()
    # risk = StopLossRiskManager(stop_loss_pct=0.02)
    # risk = MaxExposureRiskManager(max_exposure_pct=0.30)
    # risk = DrawdownLimitRiskManager(max_drawdown_pct=0.10)
    # risk = VixFilterRiskManager(daily_loss_limit=0.03, vol_threshold=0.005)
    risk = PassThroughRiskManager()

    run_backtest(model, sizer, risk)


if __name__ == "__main__":
    main()
