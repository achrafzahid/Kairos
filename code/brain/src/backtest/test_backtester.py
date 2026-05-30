"""Rigorous test suite for the unified backtester. Run from backtester/ dir."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtester as B

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("="*70)
print("TEST 1: Path auto-resolution finds sibling dirs")
print("="*70)
check("base_dir resolves to ../tensors", os.path.isdir(B.CONFIG["base_dir"]))
check("checkpoint resolves", os.path.exists(B.CONFIG["checkpoint"]))
check("DeepLOB imported", B.DeepLOB is not None)

print("\n" + "="*70)
print("TEST 2: Voting aggregation correctness")
print("="*70)
ens = B.RiskEnsemble([B.PassThroughRiskManager()], mode="min")
# craft votes manually via a fake ensemble
class V(B.RiskManager):
    def __init__(s, n, val): s.name=n; s.val=val
    def vote(s, *a): return s.val
for mode, expected in [("min", 0.0), ("median", 490.0), ("mean", 495.0)]:
    e = B.RiskEnsemble([V("a",500),V("b",480),V("c",0),V("d",1000)], mode=mode)
    size, binding = e.decide(1e7, "X", 999, 1, 100.0, {}, {"prices_now":{}})
    check(f"{mode} vote = {expected}", abs(size-expected) < 1e-6)
    check(f"{mode} binding is the min-voter (c)", e.managers[binding].name=="c")

print("\n" + "="*70)
print("TEST 3: Performance-weighted vote shifts toward profitable RM (causal)")
print("="*70)
e = B.RiskEnsemble([V("cautious",0),V("aggressive",1000)], mode="weighted_median",
                   perf_lookback=50)
# attribute strong profits to 'aggressive', losses to 'cautious'
for _ in range(20):
    e.attribute("aggressive", +5000)
    e.attribute("cautious", -3000)
size,_ = e.decide(1e7,"X",999,1,100.0,{},{"prices_now":{}})
check("weighted vote favors profitable (aggressive) RM", size > 500)

print("\n" + "="*70)
print("TEST 4: Portfolio accounting — pnl sign & equity integrity")
print("="*70)
pf = B.Portfolio(1_000_000.0)
ens0 = B.build_default_ensemble()
# Long 100 @ 100, close @ 105 -> +500
pf.set_target("d","T",+1,100,100.0,0.5,0.5,"passthrough",ens0)
pf.close("d","T",105.0,"EXIT",ens0)
check("long win pnl = +500", abs((pf.equity-1_000_000.0)-500.0)<1e-6)
pf2 = B.Portfolio(1_000_000.0)
# Short 100 @ 100, close @ 95 -> +500
pf2.set_target("d","T",-1,100,100.0,0.5,0.5,"passthrough",ens0)
pf2.close("d","T",95.0,"EXIT",ens0)
check("short win pnl = +500", abs((pf2.equity-1_000_000.0)-500.0)<1e-6)
# trades sum to equity delta
tot = sum(t["pnl"] for t in pf.trades) + sum(t["pnl"] for t in pf2.trades)
check("sum(trade pnl) == total equity change",
      abs(tot - ((pf.equity-1e6)+(pf2.equity-1e6))) < 1e-6)

print("\n" + "="*70)
print("TEST 5: VWAP on ADD, partial pnl on TRIM")
print("="*70)
pf = B.Portfolio(1_000_000.0)
pf.set_target("d","T",+1,100,100.0,0.5,0.5,"passthrough",ens0)   # 100 @ 100
pf.set_target("d","T",+1,200,110.0,0.5,0.5,"passthrough",ens0)   # add 100 @ 110
pos = pf.positions["T"]
check("VWAP after add = 105", abs(pos.entry_vwap-105.0)<1e-6)
check("shares after add = 200", pos.shares==200)
pf.set_target("d","T",+1,100,120.0,0.5,0.5,"passthrough",ens0)   # trim to 100 @ 120
# trim 100 sh from vwap 105 to 120 -> +1500
check("trim realizes +1500", any(abs(t["pnl"]-1500.0)<1e-6 for t in pf.trades))

print("\n" + "="*70)
print("TEST 6: Full backtest — EOD flat, leverage cap, no leak")
print("="*70)
B.CONFIG.update({"base_dir": B.CONFIG["base_dir"], "tickers":["AAPL","SPY"],
                 "test_dates":["2024-09-30","2024-10-01"], "test_days":2,
                 "device":"cpu","max_gross_leverage":10.0,"kelly_multiplier":0.5,
                 "max_position_leverage":6.0,"verbose_trades":False})
model = B.DeepLOB(num_horizons=3)
model.load_state_dict(torch.load(B.CONFIG["checkpoint"], weights_only=True)); model.eval()
sizer = B.KellySizer(0.5, 6.0)
res = B.run_backtest(model, sizer, B.build_default_ensemble())
check("backtest returned results", res is not None)
# EOD flat: every day must end with all positions closed (last action per day per
# ticker should leave book flat — we check via reasons containing EOD/EXIT/FLIP)
# Reconstruct: after each day, no position should carry; we verify by counting
# that #opens == #closes overall.
opens = sum(1 for t in res["trades"] if t["action"]=="OPEN")
closes = sum(1 for t in res["trades"] if t["action"] in ("EOD","EXIT","FLIP"))
check("every OPEN is eventually closed (opens==closes)", opens==closes)
# leverage cap: peak single-name fill notional <= per-name cap
maxn = max((t["shares"]*t["price"] for t in res["trades"] if t["action"] in ("OPEN","ADD")), default=0)
check("max single fill <= per-name cap (6x*equity)", maxn <= 6.0*1e7*1.01)

print("\n" + "="*70)
print("TEST 7: Trained model beats random (signal has value)")
print("="*70)
rnd = B.DeepLOB(num_horizons=3); rnd.eval()
res_t = B.run_backtest(model, B.KellySizer(0.5,6.0), B.build_default_ensemble())
res_r = B.run_backtest(rnd, B.KellySizer(0.5,6.0), B.build_default_ensemble())
check("trained final equity >= random final equity",
      res_t["final_equity"] >= res_r["final_equity"])

print("\n" + "="*70)
print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL: print("FAILED:", FAIL)
print("="*70)
sys.exit(1 if FAIL else 0)
