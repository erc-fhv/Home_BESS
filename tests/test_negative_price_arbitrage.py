import pandas as pd

from control.optimize import BessOptimizer


def test_optimize_milp_blocks_simultaneous_grid_arbitrage():
    day = pd.Timestamp("2025-01-01", tz="Europe/Vienna")
    act_range = pd.date_range(
        start=day,
        end=day + pd.DateOffset(days=1),
        freq="15min",
        tz="Europe/Vienna",
        inclusive="left",
    )

    price_buy = pd.Series(0.20, index=act_range)
    price_sell = pd.Series(0.05, index=act_range)
    price_sell.iloc[56] = 0.25
    net_load = pd.Series(0.0, index=act_range)

    result = BessOptimizer().optimize_milp(
        price_sell_eur_kwh=price_sell,
        price_buy_eur_kwh=price_buy,
        net_load_kw=net_load,
        soc_init_percent=50.0,
        soc_final_percent=50.0,
        capacity_kwh=30.72,
        max_charge_kw=8.0,
        max_discharge_kw=8.0,
        soc_min_percent=10.0,
        eta_charge=0.936,
        eta_discharge=0.936,
        allow_battery_feed_in=True,
        objective="profit",
    )

    simultaneous_grid_flow = (result["p_buy_kw"] > 1e-9) & (result["p_sell_kw"] > 1e-9)

    assert (price_sell > price_buy).any()
    assert result["milp_status"] == "Optimal"
    assert not simultaneous_grid_flow.any()