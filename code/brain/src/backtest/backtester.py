"""
backtester.py — Unified DeepLOB trading simulator (quant-grade).

Merges the two prior backtests into one engine:
  - Multi-horizon edge fusion (P_up - P_down across h10/h20/h50), from v2.
  - Full sizing library (Fixed, Confidence, Volatility, Kelly), from v1,
    plus an adaptive "blend" sizer that auto-selects causally.
  - Dynamic position management: OPEN / ADD / TRIM / FLIP / EXIT, from v2.
  - RISK-MANAGER ENSEMBLE VOTE: every risk manager independently votes on the
    size of each proposed trade; votes are aggregated by performance-weighted
    consensus (each RM weighted by how well its own past vetoes/sizes performed).
    No look-ahead: weights use only already-closed trades.
  - Hard book-level gross-leverage backstop on top of the vote.
  - Books cleared every end-of-day. Fees = 0 (configurable).

Directory layout (run from a sibling dir of ETL/ and DeepLOB/):
    project/
      ETL/         (transform.py, load.py, ...)
      DeepLOB/     (deeplob.py, main.py, best_deeplob_model.pth)
      backtester/  (THIS FILE)
      tensors/     (AAPL/.., SPY/.., NVDA/..)  <- written by the ETL
backtester resolves these relative paths automatically; override in CONFIG.

Leak-free by construction: decision at window i uses only X[i] (the 100 past
snapshots). Execution uses mid[keep[i] + entry_delay], a price at/after the
decision, never the future mid the label came from.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F

# ---- Make sibling DeepLOB/ importable for the model definition ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
for _cand in (os.path.join(_PROJECT, "DeepLOB"), os.path.join(_PROJECT, "deeplob"),
              _PROJECT):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from deeplob import DeepLOB
except Exception as e:  # pragma: no cover - only triggers on misconfigured paths
    DeepLOB = None
    _IMPORT_ERR = e

HORIZONS = [10, 20, 50]
N_TIMESTEPS = 100


# ============================================================================
# CONFIG
# ============================================================================
def _default_base_dir():
    for cand in (os.path.join(_PROJECT, "tensors"),
                 os.path.join(_HERE, "..", "tensors"),
                 "../tensors"):
        if os.path.isdir(cand):
            return cand
    return os.path.join(_PROJECT, "tensors")


def _default_ckpt():
    for cand in (os.path.join(_PROJECT, "DeepLOB", "best_deeplob_model.pth"),
                 os.path.join(_PROJECT, "DeepLOB", "last_deeplob_model.pth"),
                 "best_deeplob_model.pth"):
        if os.path.exists(cand):
            return cand
    return os.path.join(_PROJECT, "DeepLOB", "best_deeplob_model.pth")


CONFIG = {
    "base_dir": _default_base_dir(),
    "tickers": ["AAPL", "SPY", "NVDA"],
    "checkpoint": _default_ckpt(),
    "initial_capital": 10_000_000.0,

    "test_days": 10,
    "test_dates": None,                 # pin specific ['YYYY-MM-DD', ...]

    # Aggression / sizing
    "max_gross_leverage": 10.0,         # hard cap: total notional <= Nx equity
    "kelly_multiplier": 0.5,            # fractional Kelly on edge (0.5 = half)
    "max_position_leverage": 6.0,       # per-name notional cap (x equity)
    "min_edge_to_trade": 0.05,
    "add_threshold": 0.10,
    "trim_threshold": 0.10,
    "entry_delay": 1,

    # Horizon fusion
    "horizon_weights": [0.2, 0.3, 0.5],

    # Risk-manager ensemble
    "vote_mode": "weighted_median",     # weighted_median | median | min | mean
    "vote_perf_lookback": 50,           # closed trades used to weight each RM
    "fees_per_share": 0.0,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose_trades": False,
}


# ============================================================================
# SIZING LIBRARY  (target SIGNED notional in $)
# ============================================================================
class Sizer:
    def target_notional(self, equity, edge, side, conf, ctx):
        raise NotImplementedError


class KellySizer(Sizer):
    """Fractional Kelly on the model's signed edge in [-1, 1]."""
    def __init__(self, kelly_mult=0.5, max_pos_lev=6.0):
        self.k = kelly_mult
        self.max_pos_lev = max_pos_lev

    def target_notional(self, equity, edge, side, conf, ctx):
        frac = min(self.k * abs(edge), 1.0)
        return side * frac * equity * self.max_pos_lev


class FixedFractionSizer(Sizer):
    def __init__(self, fraction=0.10, max_pos_lev=1.0):
        self.fraction = fraction
        self.max_pos_lev = max_pos_lev

    def target_notional(self, equity, edge, side, conf, ctx):
        return side * self.fraction * equity * self.max_pos_lev


class ConfidenceWeightedSizer(Sizer):
    """Linear ramp in |edge| up to a per-name leverage budget."""
    def __init__(self, max_pos_lev=6.0, floor=0.0):
        self.max_pos_lev = max_pos_lev
        self.floor = floor

    def target_notional(self, equity, edge, side, conf, ctx):
        ramp = max(0.0, abs(edge) - self.floor) / max(1e-9, 1.0 - self.floor)
        return side * ramp * equity * self.max_pos_lev


class VolatilityScaledSizer(Sizer):
    """Kelly-like base, divided by trailing mid volatility (risk parity-ish)."""
    def __init__(self, kelly_mult=0.5, max_pos_lev=6.0, lookback=50, vol_scale=200.0):
        self.k = kelly_mult
        self.max_pos_lev = max_pos_lev
        self.lookback = lookback
        self.vol_scale = vol_scale

    def target_notional(self, equity, edge, side, conf, ctx):
        base = min(self.k * abs(edge), 1.0) * equity * self.max_pos_lev
        recent = ctx.get("recent_mids", [])
        if len(recent) >= self.lookback:
            r = np.diff(recent[-self.lookback:]) / np.asarray(recent[-self.lookback:-1])
            vol = float(np.std(r))
            base *= 1.0 / (1.0 + self.vol_scale * vol)
        return side * base


# ============================================================================
# RISK MANAGERS  (each VOTES a size in shares; engine aggregates the votes)
# ============================================================================
class RiskManager:
    name = "base"

    def vote(self, equity, ticker, proposed_shares, side, price,
             open_positions, ctx):
        """Return the share count THIS manager would allow (0 = veto)."""
        raise NotImplementedError


class PassThroughRiskManager(RiskManager):
    name = "passthrough"

    def vote(self, equity, ticker, proposed_shares, side, price,
             open_positions, ctx):
        return proposed_shares


class StopLossRiskManager(RiskManager):
    """Veto adding when the current position is underwater beyond a threshold."""
    name = "stoploss"

    def __init__(self, stop_loss_pct=0.02):
        self.stop_loss_pct = stop_loss_pct

    def vote(self, equity, ticker, proposed_shares, side, price,
             open_positions, ctx):
        pos = open_positions.get(ticker)
        if pos is not None:
            pnl_pct = pos.side * (price - pos.entry_vwap) / pos.entry_vwap
            if pnl_pct <= -self.stop_loss_pct:
                return 0.0
        return proposed_shares


class MaxExposureRiskManager(RiskManager):
    """Cap total gross notional across names at a fraction*leverage of equity."""
    name = "maxexposure"

    def __init__(self, max_exposure_x=8.0):
        self.max_x = max_exposure_x

    def vote(self, equity, ticker, proposed_shares, side, price,
             open_positions, ctx):
        cur = sum(p.shares * ctx["prices_now"].get(t, p.entry_vwap)
                  for t, p in open_positions.items() if t != ticker)
        room = max(0.0, self.max_x * equity - cur)
        return min(proposed_shares, room / price) if price > 0 else 0.0


class DrawdownLimitRiskManager(RiskManager):
    """Throttle size as equity falls below trailing peak; veto past a hard limit."""
    name = "drawdown"

    def __init__(self, max_drawdown_pct=0.10):
        self.max_dd = max_drawdown_pct
        self.peak = None

    def vote(self, equity, ticker, proposed_shares, side, price,
             open_positions, ctx):
        self.peak = equity if self.peak is None else max(self.peak, equity)
        dd = (self.peak - equity) / self.peak if self.peak > 0 else 0.0
        if dd >= self.max_dd:
            return 0.0
        return proposed_shares * (1.0 - dd / self.max_dd)   # linear throttle


class VolatilityRiskManager(RiskManager):
    """Shrink size when trailing mid volatility is high."""
    name = "volatility"

    def __init__(self, vol_lookback=30, vol_scale=300.0):
        self.lb = vol_lookback
        self.scale = vol_scale

    def vote(self, equity, ticker, proposed_shares, side, price,
             open_positions, ctx):
        recent = ctx.get("recent_mids", [])
        if len(recent) <= self.lb:
            return proposed_shares
        r = np.diff(recent[-self.lb:]) / np.asarray(recent[-self.lb:-1])
        vol = float(np.std(r))
        return proposed_shares * (1.0 / (1.0 + self.scale * vol))


# ============================================================================
# RISK ENSEMBLE — performance-weighted causal voting
# ============================================================================
class RiskEnsemble:
    """Each manager votes a size; we aggregate. In 'weighted_median' mode, each
    manager's weight is its trailing realized-PnL performance attribution: how
    well trades have done when that manager's vote was the binding (smallest)
    one. Weights use only CLOSED trades -> no look-ahead."""

    def __init__(self, managers, mode="weighted_median", perf_lookback=50):
        self.managers = managers
        self.mode = mode
        self.perf_lookback = perf_lookback
        # rolling realized pnl attributed to each manager when it was binding
        self.attr = {m.name: [] for m in managers}

    def _weights(self):
        w = []
        for m in self.managers:
            hist = self.attr[m.name][-self.perf_lookback:]
            if not hist:
                w.append(1.0)                 # neutral prior
            else:
                avg = float(np.mean(hist))
                # map avg pnl -> positive weight via softplus-ish; reward profit
                w.append(max(1e-3, np.log1p(np.exp(avg / (abs(avg) + 1e-9) * 2))))
        w = np.asarray(w, dtype=float)
        return w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)

    def decide(self, *vote_args):
        votes = np.array([m.vote(*vote_args) for m in self.managers], dtype=float)
        votes = np.clip(votes, 0.0, None)
        binding = int(np.argmin(votes))       # the manager constraining the size
        if self.mode == "min":
            size = votes.min()
        elif self.mode == "mean":
            size = votes.mean()
        elif self.mode == "median":
            size = float(np.median(votes))
        else:  # weighted_median
            w = self._weights()
            order = np.argsort(votes)
            v, wv = votes[order], w[order]
            cum = np.cumsum(wv) / wv.sum()
            size = float(v[min(int(np.searchsorted(cum, 0.5)), len(v) - 1)])
        return size, binding

    def attribute(self, binding_name, realized_pnl):
        """Record realized pnl against whichever manager was binding at entry."""
        if binding_name in self.attr:
            self.attr[binding_name].append(realized_pnl)


# ============================================================================
# PORTFOLIO
# ============================================================================
class Position:
    __slots__ = ("side", "entry_vwap", "shares", "binding")

    def __init__(self, side, entry_vwap, shares, binding=None):
        self.side = side
        self.entry_vwap = entry_vwap
        self.shares = shares
        self.binding = binding      # name of the risk manager that sized entry


class Portfolio:
    def __init__(self, initial_capital, fees_per_share=0.0):
        self.equity = float(initial_capital)
        self.fees = float(fees_per_share)
        self.positions = {}
        self.trades = []

    def is_flat(self, ticker):
        return ticker not in self.positions

    def _fee(self, shares):
        return self.fees * shares

    def _record(self, day, ticker, action, side, shares, price, pnl, conf, edge):
        self.trades.append({
            "day": day, "ticker": ticker, "action": action,
            "side": "long" if side > 0 else "short",
            "shares": shares, "price": price, "pnl": pnl,
            "conf": conf, "edge": edge,
        })
        if CONFIG["verbose_trades"]:
            print(f"      [{day} {ticker}] {action:5s} "
                  f"{'L' if side > 0 else 'S'} {shares:9.0f} @ {price:.4f} "
                  f"pnl={pnl:+.2f} edge={edge:+.2f}")

    def set_target(self, day, ticker, side, target_shares, price, conf, edge,
                   binding, ensemble):
        cur = self.positions.get(ticker)
        if cur is None:
            if target_shares > 0:
                fee = self._fee(target_shares)
                self.equity -= fee
                self.positions[ticker] = Position(side, price, target_shares, binding)
                self._record(day, ticker, "OPEN", side, target_shares, price, -fee, conf, edge)
            return

        if side != cur.side:
            self._close(day, ticker, price, conf, edge, "FLIP", ensemble)
            if target_shares > 0:
                fee = self._fee(target_shares)
                self.equity -= fee
                self.positions[ticker] = Position(side, price, target_shares, binding)
                self._record(day, ticker, "OPEN", side, target_shares, price, -fee, conf, edge)
            return

        delta = target_shares - cur.shares
        if delta > 0:                      # ADD
            fee = self._fee(delta)
            self.equity -= fee
            new_total = cur.shares + delta
            cur.entry_vwap = (cur.entry_vwap * cur.shares + price * delta) / new_total
            cur.shares = new_total
            self._record(day, ticker, "ADD", side, delta, price, -fee, conf, edge)
        elif delta < 0:                    # TRIM
            qty = -delta
            fee = self._fee(qty)
            pnl = cur.side * (price - cur.entry_vwap) * qty - fee
            self.equity += pnl
            cur.shares -= qty
            self._record(day, ticker, "TRIM", side, qty, price, pnl, conf, edge)
            if ensemble is not None and cur.binding is not None:
                ensemble.attribute(cur.binding, pnl)
            if cur.shares <= 1e-9:
                self.positions.pop(ticker, None)

    def _close(self, day, ticker, price, conf, edge, reason, ensemble):
        pos = self.positions.pop(ticker)
        fee = self._fee(pos.shares)
        pnl = pos.side * (price - pos.entry_vwap) * pos.shares - fee
        self.equity += pnl
        self._record(day, ticker, reason, pos.side, pos.shares, price, pnl, conf, edge)
        if ensemble is not None and pos.binding is not None:
            ensemble.attribute(pos.binding, pnl)
        return pnl

    def close(self, day, ticker, price, reason="EOD", ensemble=None):
        if ticker in self.positions:
            return self._close(day, ticker, price, 0.0, 0.0, reason, ensemble)
        return 0.0


# ============================================================================
# ALIGNMENT
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
# PREDICTION
# ============================================================================
@torch.no_grad()
def predict_day(model, X, device, batch=512):
    w = np.array(CONFIG["horizon_weights"], dtype=np.float64)
    w = w / w.sum()
    edges, confs, sides = [], [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        probs = F.softmax(model(xb), dim=2).cpu().numpy()
        e_h = probs[:, :, 1] - probs[:, :, 2]
        fused = e_h @ w
        edges.append(fused)
        confs.append(np.abs(fused))
        sides.append(np.sign(fused))
    return (np.concatenate(edges), np.concatenate(confs),
            np.concatenate(sides).astype(int))


# ============================================================================
# SIMULATE ONE TICKER-DAY
# ============================================================================
def simulate_ticker_day(model, pf, paths, ticker, day, sizer, ensemble,
                        device, prices_now):
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
    add_thr, trim_thr = CONFIG["add_threshold"], CONFIG["trim_threshold"]

    for i in range(n):
        ei = i + delay
        if ei >= n:
            break
        price = float(mid_exec[ei])
        if not np.isfinite(price) or price <= 0:
            continue
        prices_now[ticker] = price
        e, s = float(edge[i]), int(side[i])

        if abs(e) < min_edge or s == 0:
            if ticker in pf.positions:
                pf.close(day, ticker, price, "EXIT", ensemble)
            continue

        # 1) Sizer proposes a target notional.
        tgt_notional = sizer.target_notional(
            pf.equity, e, s, conf[i],
            ctx={"ticker": ticker,
                 "recent_mids": mid_exec[max(0, i - 100): i + 1].tolist()})
        proposed_shares = abs(tgt_notional) / price

        # 2) Risk ensemble votes on the size (causal, perf-weighted).
        rm_ctx = {"ticker": ticker, "day": day, "prices_now": prices_now,
                  "recent_mids": mid_exec[max(0, i - 100): i + 1].tolist()}
        voted_shares, binding = ensemble.decide(
            pf.equity, ticker, proposed_shares, s, price, pf.positions, rm_ctx)
        binding_name = ensemble.managers[binding].name

        # 3) Hard book-level gross-leverage backstop.
        gross_cap = CONFIG["max_gross_leverage"] * pf.equity
        gross_others = sum(p.shares * prices_now.get(t, p.entry_vwap)
                           for t, p in pf.positions.items() if t != ticker)
        room = max(0.0, gross_cap - gross_others)
        voted_shares = min(voted_shares, room / price) if price > 0 else 0.0
        target_shares = float(int(voted_shares))

        cur = pf.positions.get(ticker)
        if cur is None:
            if target_shares > 0:
                pf.set_target(day, ticker, s, target_shares, price,
                              conf[i], e, binding_name, ensemble)
        elif s != cur.side:
            pf.set_target(day, ticker, s, target_shares, price,
                          conf[i], e, binding_name, ensemble)
        else:
            cur_n = cur.shares * price
            want_n = target_shares * price
            rel = (want_n - cur_n) / max(cur_n, 1.0)
            if rel >= add_thr or rel <= -trim_thr:
                pf.set_target(day, ticker, s, target_shares, price,
                              conf[i], e, binding_name, ensemble)

    if ticker in pf.positions:
        pf.close(day, ticker, float(mid_exec[-1]), "EOD", ensemble)
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


def _metrics(pf, daily, init_cap):
    total = pf.equity - init_cap
    tret = 100.0 * total / init_cap
    rets = np.array([d[4] / 100.0 for d in daily], dtype=float)
    sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(252)) if rets.std() > 0 else 0.0
    closing = [t for t in pf.trades if t["pnl"] != 0.0]
    wins = [t for t in closing if t["pnl"] > 0]
    wr = 100.0 * len(wins) / len(closing) if closing else 0.0
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in closing if t["pnl"] < 0)
    pf_ratio = (gp / gl) if gl > 0 else float("inf")
    peak, mdd = init_cap, 0.0
    eq = init_cap
    for _, dp, e, _, _ in daily:
        peak = max(peak, e)
        mdd = min(mdd, 100.0 * (e - peak) / peak)
    return {"total": total, "tret": tret, "sharpe": sharpe, "win_rate": wr,
            "profit_factor": pf_ratio, "max_dd": mdd, "n_closing": len(closing)}


def run_backtest(model, sizer, ensemble):
    device = CONFIG["device"]
    base, tickers = CONFIG["base_dir"], CONFIG["tickers"]
    dates = discover_test_dates(base, tickers, CONFIG["test_days"], CONFIG["test_dates"])
    if not dates:
        print(f"No test dates under {base}/{tickers[0]}.")
        return None

    pf = Portfolio(CONFIG["initial_capital"], CONFIG["fees_per_share"])
    print(f"\n{'=' * 80}")
    print(f"BACKTEST | ${pf.equity:,.0f} | gross<= {CONFIG['max_gross_leverage']}x "
          f"| sizer={sizer.__class__.__name__} | vote={CONFIG['vote_mode']} "
          f"| RMs={[m.name for m in ensemble.managers]}")
    print(f"{'=' * 80}\nTest days: {dates}\n")

    daily = []
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
                                sizer, ensemble, device, prices_now)
        day_pnl = pf.equity - eq_before
        ntr = len(pf.trades) - tr_before
        ret = 100.0 * day_pnl / eq_before if eq_before else 0.0
        daily.append((day, day_pnl, pf.equity, ntr, ret))
        print(f"{day} | P&L {day_pnl:+13,.2f} | equity ${pf.equity:15,.2f} "
              f"| {ntr:5d} fills | {ret:+.3f}%")

    m = _metrics(pf, daily, CONFIG["initial_capital"])
    print(f"\n{'-' * 80}")
    print(f"TOTAL P&L {m['total']:+,.2f} | final ${pf.equity:,.2f} "
          f"| return {m['tret']:+.3f}% | Sharpe(ann) {m['sharpe']:.2f} "
          f"| max DD {m['max_dd']:.2f}%")
    print(f"Realizing trades {m['n_closing']} | win {m['win_rate']:.1f}% "
          f"| profit factor {m['profit_factor']:.2f}")
    if CONFIG["vote_mode"] == "weighted_median":
        wnorm = ensemble._weights()
        print("RM trailing weights: " +
              "  ".join(f"{mn.name}:{w:.2f}"
                        for mn, w in zip(ensemble.managers, wnorm)))
    print(f"{'-' * 80}\n")
    return {"daily": daily, "trades": pf.trades, "final_equity": pf.equity, **m}


def build_default_ensemble():
    return RiskEnsemble(
        managers=[
            PassThroughRiskManager(),
            StopLossRiskManager(stop_loss_pct=0.02),
            MaxExposureRiskManager(max_exposure_x=8.0),
            DrawdownLimitRiskManager(max_drawdown_pct=0.10),
            VolatilityRiskManager(vol_lookback=30, vol_scale=300.0),
        ],
        mode=CONFIG["vote_mode"],
        perf_lookback=CONFIG["vote_perf_lookback"],
    )


def main():
    if DeepLOB is None:
        print(f"Could not import DeepLOB. Check that DeepLOB/deeplob.py exists. "
              f"({_IMPORT_ERR})")
        return
    device = CONFIG["device"]
    model = DeepLOB(num_horizons=len(HORIZONS)).to(device)
    ckpt = CONFIG["checkpoint"]
    if not os.path.exists(ckpt):
        print(f"Checkpoint '{ckpt}' not found. Train first or set CONFIG['checkpoint'].")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {ckpt} on {device}")

    sizer = KellySizer(kelly_mult=CONFIG["kelly_multiplier"],
                       max_pos_lev=CONFIG["max_position_leverage"])
    ensemble = build_default_ensemble()
    run_backtest(model, sizer, ensemble)


if __name__ == "__main__":
    main()
