from pathlib import Path
from flask import config
import pulp
import numpy as np
import pandas as pd

class BessOptimizer:

    def optimize_milp(
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
        allow_battery_feed_in: bool = True,
        objective: str = "profit",
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
        soc_min_kwh = soc_min_percent * capacity_kwh / 100.0
        soc_max_kwh = soc_max_percent * capacity_kwh / 100.0
        soc_kwh    = pulp.LpVariable.dicts("soc", T, soc_min_kwh, soc_max_kwh)
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

        # Batterie-Einspeisung verbieten: Verkauf nur aus (vorhergesagtem) PV-Ueberschuss
        if not allow_battery_feed_in:
            for p in P:
                pv_surplus = max(0.0, -float(net_load_kw.iloc[p]))
                model += p_sell_kw[p] <= pv_surplus

        # Zielfunktion
        if objective == "autarky":
            # Maximiere Autarkie = minimiere Netzbezug
            model += -pulp.lpSum(p_buy_kw[p] * delta_t for p in P)
        elif objective == "peak_shaving":
            # Minimiere maximale Netzspitze (Bezug UND Einspeisung)
            p_peak = pulp.LpVariable("p_peak", 0)
            for p in P:
                model += p_buy_kw[p] - p_sell_kw[p] <=  p_peak   # Bezugsspitze
                model += p_sell_kw[p] - p_buy_kw[p] <=  p_peak   # Einspeisespitze
            model += -p_peak
        else:
            # Profit: Erlös – Kosten
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
            "profit_eur": float(sum(
                (price_sell_eur_kwh.iloc[p] * (p_sell_kw[p].value() or 0.0)
                 - price_buy_eur_kwh.iloc[p] * (p_buy_kw[p].value() or 0.0)) * delta_t
                for p in P
            )),
            "p_buy_kw": pd.Series([p_buy_kw[p].value() for p in P], index=time_periods),
            "p_sell_kw": pd.Series([p_sell_kw[p].value() for p in P], index=time_periods),
        }

        return optimization_results

    def no_optimize(
        self,
        price_sell_eur_kwh: pd.Series,
        price_buy_eur_kwh: pd.Series,
        net_load_kw: pd.Series,
        verbose: bool = False,
    ) -> dict:
        """Ohne Batterie: Netzlast geht direkt ins Netz."""

        time_points = net_load_kw.index
        time_periods = time_points[:-1]
        delta_t = (time_points[1] - time_points[0]).total_seconds() / 3600.0

        nl = net_load_kw.iloc[:-1]
        p_buy = nl.clip(lower=0)
        p_buy.index = time_periods
        p_sell = (-nl).clip(lower=0)
        p_sell.index = time_periods
        profit = float((price_sell_eur_kwh.iloc[:-1] * p_sell - price_buy_eur_kwh.iloc[:-1] * p_buy).sum() * delta_t)

        return {
            "soc_percent": pd.Series(0.0, index=time_points),
            "p_ch_kw": pd.Series(0.0, index=time_periods),
            "p_dis_kw": pd.Series(0.0, index=time_periods),
            "set_netload_kw": pd.Series(nl.values, index=time_periods),
            "milp_status": "No battery",
            "date": pd.Timestamp(time_points[0]).normalize().date(),
            "profit_eur": profit,
            "p_buy_kw": p_buy,
            "p_sell_kw": p_sell,
        }

    def pv_surplus_charge(
        self,
        price_sell_eur_kwh: pd.Series,
        price_buy_eur_kwh: pd.Series,
        net_load_kw: pd.Series,
        soc_init_percent: float,
        capacity_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        soc_min_percent: float,
        soc_max_percent: float,
        eta_charge: float,
        eta_discharge: float,
        verbose: bool = False,
    ) -> dict:
        """PV-Ueberschussladen: Ueberschuss laden, Bedarf aus Batterie decken."""

        time_points = net_load_kw.index
        time_periods = time_points[:-1]
        nl = net_load_kw.iloc[:-1].values  # 95 periods, like MILP
        n = len(nl)
        delta_t = (time_points[1] - time_points[0]).total_seconds() / 3600.0

        soc_min_kwh = soc_min_percent * capacity_kwh / 100.0
        soc_max_kwh = soc_max_percent * capacity_kwh / 100.0

        # Pre-allocate arrays
        soc = np.empty(n + 1)
        p_ch = np.zeros(n)
        p_dis = np.zeros(n)

        soc[0] = soc_init_percent * capacity_kwh / 100.0

        for i in range(n):
            if nl[i] < 0:
                # PV-Ueberschuss → Batterie laden
                surplus = -nl[i]
                charge_room = soc_max_kwh - soc[i]
                max_ch = min(max_charge_kw,
                             charge_room / (eta_charge * delta_t)) \
                    if eta_charge * delta_t > 0 else 0.0
                p_ch[i] = min(surplus, max(max_ch, 0.0))
                soc[i + 1] = soc[i] + p_ch[i] * eta_charge * delta_t
            else:
                # Verbrauch → Batterie entladen
                discharge_room = soc[i] - soc_min_kwh
                max_dis = min(max_discharge_kw,
                              discharge_room * eta_discharge / delta_t) \
                    if delta_t > 0 else 0.0
                p_dis[i] = min(nl[i], max(max_dis, 0.0))
                soc[i + 1] = soc[i] - p_dis[i] / eta_discharge * delta_t

        # Leistungsbilanz
        p_buy = np.maximum(nl - p_dis + p_ch, 0.0)   # Netzbezug (abzgl. Entladung, zzgl. Ladung)
        p_sell = np.maximum(-(nl - p_dis + p_ch), 0.0) # Einspeisung

        # Netz-Residuallast = Bezug - Einspeisung = nl + p_ch - p_dis
        set_netload = p_buy - p_sell

        soc_percent = soc / capacity_kwh * 100.0

        # Profit
        sell_prices = price_sell_eur_kwh.iloc[:-1].values
        buy_prices = price_buy_eur_kwh.iloc[:-1].values
        profit = float(np.sum((sell_prices * p_sell - buy_prices * p_buy) * delta_t))

        return {
            "soc_percent": pd.Series(soc_percent, index=time_points),
            "p_ch_kw": pd.Series(p_ch, index=time_periods),
            "p_dis_kw": pd.Series(p_dis, index=time_periods),
            "set_netload_kw": pd.Series(set_netload, index=time_periods),
            "milp_status": "PV surplus",
            "date": pd.Timestamp(time_points[0]).normalize().date(),
            "profit_eur": profit,
            "p_buy_kw": pd.Series(p_buy, index=time_periods),
            "p_sell_kw": pd.Series(p_sell, index=time_periods),
        }
