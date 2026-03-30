"""Tests for BessOptimizer.pv_surplus_charge using the default net load
profile for 2025-11-05 (autumn day with midday PV surplus)."""

import numpy as np
import pandas as pd
from pathlib import Path

from control.optimize import BessOptimizer


def test_pv_surplus_charge():
    """Core behavioral test on the real 2025-11-05 net-load profile.

    Checks three invariants of PV surplus charging:
    1. Negative net load (PV surplus) must cause SOC to rise.
    2. Grid feed-in (p_sell > 0) only occurs when the battery is full.
    3. Grid purchase (p_buy > 0) only occurs when the battery is empty.
    """
    # -- setup: load real net-load data and run algorithm --
    data_file = Path(__file__).parent / ".." / "src" / "simulation" / "data" \
        / "example_household_without_battery.csv"
    netload_df = pd.read_csv(data_file, index_col=0)
    netload_df.index = pd.to_datetime(netload_df.index, utc=True).tz_convert("Europe/Vienna")

    day = pd.Timestamp("2025-11-05", tz="Europe/Vienna")
    capacity_kwh = 10.0
    soc_min = 10.0
    soc_max = 90.0
    tol = 1e-9

    act_range = pd.date_range(
        start=day, end=day + pd.DateOffset(days=1),
        freq="15min", tz="Europe/Vienna", inclusive="left",
    )
    net_load_kw = netload_df.loc[act_range]["net_load_kw"]

    print(net_load_kw, flush=True)

    r = BessOptimizer().pv_surplus_charge(
        price_sell_eur_kwh=pd.Series(0.05, index=act_range),
        price_buy_eur_kwh=pd.Series(0.25, index=act_range),
        net_load_kw=net_load_kw,
        soc_init_percent=50.0,
        capacity_kwh=capacity_kwh,
        max_charge_kw=5.0,
        max_discharge_kw=5.0,
        soc_min_percent=soc_min,
        soc_max_percent=soc_max,
        eta_charge=0.95,
        eta_discharge=0.95,
    )

    soc = r["soc_percent"].values          # 96 time-points
    nl = net_load_kw.iloc[:-1].values      # 95 periods
    p_sell = r["p_sell_kw"].values         # 95 periods
    p_buy = r["p_buy_kw"].values           # 95 periods

    # 1. Negative net load (PV surplus) → SOC must increase (or stay at max)
    for i in range(len(nl)):
        if nl[i] < 0:
            assert soc[i + 1] >= soc[i] - tol, \
                f"Period {i}: net_load={nl[i]:.2f} kW (surplus) but SOC dropped " \
                f"from {soc[i]:.2f}% to {soc[i+1]:.2f}%"

    # 2. Grid feed-in only when battery is full (at end of period)
    for i in range(len(nl)):
        if p_sell[i] > tol:
            assert soc[i + 1] >= soc_max - tol, \
                f"Period {i}: grid feed-in={p_sell[i]:.2f} kW but SOC_end={soc[i+1]:.2f}% " \
                f"(not full, max={soc_max}%)"

    # 3. Grid purchase only when battery is empty (at end of period)
    for i in range(len(nl)):
        if p_buy[i] > tol:
            assert soc[i + 1] <= soc_min + tol, \
                f"Period {i}: grid purchase={p_buy[i]:.2f} kW but SOC_end={soc[i+1]:.2f}% " \
                f"(not empty, min={soc_min}%)"
