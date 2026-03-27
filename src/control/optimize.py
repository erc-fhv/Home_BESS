from pathlib import Path
from flask import config
import pulp
import numpy as np
import pandas as pd

class BessOptimizer:

    def optimize(
        self,
        price_sell_eur_kwh: pd.Series,
        price_buy_eur_kwh: pd.Series,
        net_load_kw: pd.Series,
        soc_init_percent: float,
        soc_final_percent: float,
        capacity_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        soc_min_percent: float,
        eta_charge: float,
        eta_discharge: float,
        soc_max_percent: float = 100.0,
        verbose: bool = False,
        ) -> dict[str, pd.Series]:
        """
        Optimizes the BESS operation for a given day using Mixed-Integer Linear Programming (MILP).
        """

        # Konsistenzcheck
        assert price_sell_eur_kwh.index.equals(price_buy_eur_kwh.index)
        assert price_buy_eur_kwh.index.equals(net_load_kw.index)

        # Definiere Zeitindex und Parameter
        time_points = price_sell_eur_kwh.index
        T = range(len(time_points))
        time_periods = time_points[:-1]
        P = range(len(time_periods))
        delta_t = (time_points[1] - time_points[0]).total_seconds() / 3600.0  # in hours

        # Definition des Gewinnmaximierungsproblems
        model = pulp.LpProblem("DayAheadOpt", pulp.LpMaximize)

        # Entscheidungsvariablen
        p_ch_kw   = pulp.LpVariable.dicts("p_ch", P, 0, max_charge_kw)
        p_dis_kw  = pulp.LpVariable.dicts("p_dis", P, 0, max_discharge_kw)
        soc_kwh    = pulp.LpVariable.dicts("soc", T, soc_min_percent, soc_max_percent)
        p_sell_kw = pulp.LpVariable.dicts("p_sell", P, 0)
        p_buy_kw  = pulp.LpVariable.dicts("p_buy", P, 0)
        y = pulp.LpVariable.dicts("y", P, 0, 1, cat="Binary")   # Lade-/Entlade-Exklusivität

        # Anfangs- und End-SOC festsetzen
        soc_init_kwh = soc_init_percent * capacity_kwh / 100.0
        soc_final_kwh = soc_final_percent * capacity_kwh / 100.0
        model += soc_kwh[0] == soc_init_kwh
        model += soc_kwh[T[-1]] == soc_final_kwh

        # Füge SOC Nebenbedingungen hinzu
        for t in range(1, len(T)):
            model += soc_kwh[t] == soc_kwh[t-1] + \
                delta_t * (eta_charge * p_ch_kw[t-1] - p_dis_kw[t-1] / eta_discharge)

        # Lade-/Entlade-Exklusivität
        for p in P:
            model += p_ch_kw[p]  <= max_charge_kw * y[p]
            model += p_dis_kw[p] <= max_discharge_kw * (1 - y[p])

        # Leistungsbilanz
        for p in P:
            model += (
                p_buy_kw[p] + p_dis_kw[p]
                ==
                net_load_kw.iloc[p] + p_sell_kw[p] + p_ch_kw[p]
            )

        # Zielfunktion: Erlös – Kosten
        model += pulp.lpSum(
            (price_sell_eur_kwh.iloc[p] * p_sell_kw[p]
            - price_buy_eur_kwh.iloc[p] * p_buy_kw[p]) * delta_t
            for p in P
        )

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if verbose:
            print("Pulp-Resultate:")
            print(f"- Status numerisch: {model.status}")
            print(f"- Status verbal: {pulp.LpStatus[model.status]}")
            print(f"- optimaler Wert: {pulp.value(model.objective)}")

        optimization_results = {
            "soc_percent": pd.Series([soc_kwh[t].value() for t in T], index=time_points) \
                 / capacity_kwh * 100.0,
            "p_ch_kw": pd.Series([p_ch_kw[p].value() for p in P], index=time_periods),
            "p_dis_kw": pd.Series([p_dis_kw[p].value() for p in P], index=time_periods),
            "set_netload_kw": pd.Series([p_buy_kw[p].value() - p_sell_kw[p].value() for p in P],
                index=time_periods),
            "milp_status": pulp.LpStatus[model.status],
            "date": pd.Timestamp(time_points[0]).normalize().date(),
            "profit_eur": float(pulp.value(model.objective) or 0.0),
            "p_buy_kw": pd.Series([p_buy_kw[p].value() for p in P], index=time_periods),
            "p_sell_kw": pd.Series([p_sell_kw[p].value() for p in P], index=time_periods),
        }

        return optimization_results
