from pathlib import Path
from flask import config
import pulp
import numpy as np
import pandas as pd

from control.config import MpcConfig


class BessOptimizer:

    def optimize(
        self,
        my_config: MpcConfig,
        price_sell_ct_kwh: pd.Series,
        price_buy_ct_kwh: pd.Series,
        net_load_kw: pd.Series,
        soc_init_percent: float,
        verbose: bool = True,
        ) -> dict[str, pd.Series]:

        # Konsistenzcheck
        assert price_sell_ct_kwh.index.equals(price_buy_ct_kwh.index)
        assert price_buy_ct_kwh.index.equals(net_load_kw.index)

        # Definiere Zeitindex und Parameter
        time_points = price_sell_ct_kwh.index
        T = range(len(time_points))
        time_periods = time_points[:-1]
        P = range(len(time_periods))
        delta_t = (time_points[1] - time_points[0]).total_seconds() / 3600.0  # in hours

        # Definition des Gewinnmaximierungsproblems
        model = pulp.LpProblem("DayAheadOpt", pulp.LpMaximize)

        # Entscheidungsvariablen
        p_ch   = pulp.LpVariable.dicts("p_ch", P, 0, my_config.max_charge_kw)
        p_dis  = pulp.LpVariable.dicts("p_dis", P, 0, my_config.max_discharge_kw)
        soc    = pulp.LpVariable.dicts("soc", T, my_config.soc_min_percent * my_config.capacity_kwh / 100.0, my_config.capacity_kwh)
        p_sell = pulp.LpVariable.dicts("p_sell", P, 0)
        p_buy  = pulp.LpVariable.dicts("p_buy", P, 0)
        y = pulp.LpVariable.dicts("y", P, 0, 1, cat="Binary")   # Lade-/Entlade-Exklusivität

        # Anfangs- und End-SOC festsetzen
        soc_init_kwh = soc_init_percent * my_config.capacity_kwh / 100.0
        soc_final_kwh = my_config.soc_final_percent * my_config.capacity_kwh / 100.0
        model += soc[0] == soc_init_kwh
        model += soc[T[-1]] == soc_final_kwh

        # Füge SOC Nebenbedingungen hinzu
        for t in range(1, len(T)):
            model += soc[t] == soc[t-1] + \
                delta_t * (my_config.eta_charge * p_ch[t-1] - p_dis[t-1] / my_config.eta_discharge)

        # Lade-/Entlade-Exklusivität
        for p in P:
            model += p_ch[p]  <= my_config.max_charge_kw * y[p]
            model += p_dis[p] <= my_config.max_discharge_kw * (1 - y[p])

        # Leistungsbilanz
        for p in P:
            model += (
                p_buy[p] + p_dis[p]
                ==
                net_load_kw.iloc[p] + p_sell[p] + p_ch[p]
            )

        # Zielfunktion: Erlös – Kosten
        model += pulp.lpSum(
            (price_sell_ct_kwh.iloc[p] * p_sell[p]
            - price_buy_ct_kwh.iloc[p] * p_buy[p]) * delta_t
            for p in P
        )

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if verbose:
            print("Pulp-Resultate:")
            print(f"- Status numerisch: {model.status}")
            print(f"- Status verbal: {pulp.LpStatus[model.status]}")
            print(f"- optimaler Wert: {pulp.value(model.objective)}")

        optimization_results = {
            "soc":    pd.Series([soc[t].value() for t in T], index=time_points),
            "p_ch":   pd.Series([p_ch[p].value() for p in P], index=time_periods),
            "p_dis":  pd.Series([p_dis[p].value() for p in P], index=time_periods),
            "p_sell": pd.Series([p_sell[p].value() for p in P], index=time_periods),
            "p_buy":  pd.Series([p_buy[p].value() for p in P], index=time_periods),
        }

        return optimization_results
