import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Bess:
    def __init__(
        self,
        capacity_kwh: float = 5.12*4,
        max_charge_kw: float = 4.0*3,
        max_discharge_kw: float = 4.0*3,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        eta_charge: float = np.sqrt(0.95) * 0.96,       # 95% BESS round-trip efficiency
        eta_discharge: float = np.sqrt(0.95) * 0.96,    # times 96% inverter power factor
        ) -> None:

        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.soc_min_kwh = soc_min * capacity_kwh
        self.soc_max_kwh = soc_max * capacity_kwh
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge

        self.lp_result = {}

    def optimize_day(
        self,
        price_sell: pd.Series,    # ct/kWh
        price_buy: pd.Series,     # ct/kWh
        pv: pd.Series,            # kW
        load: pd.Series,          # kW
        soc0: float,              # kWh
        verbose: bool = True,
        ) -> dict:

        # Konsistenzcheck
        assert price_sell.index.equals(price_buy.index)
        assert price_buy.index.equals(pv.index)
        assert pv.index.equals(load.index)

        # Definiere Zeitindex und Parameter
        time_points = price_sell.index
        T = range(len(time_points))
        time_periods = time_points[:-1]
        P = range(len(time_periods))
        delta_t = (time_points[1] - time_points[0]).total_seconds() / 3600.0  # in hours

        # Definition des Gewinnmaximierungsproblems
        model = pulp.LpProblem("DayAheadOpt", pulp.LpMaximize)

        # Entscheidungsvariablen
        p_ch   = pulp.LpVariable.dicts("p_ch", P, 0, self.max_charge_kw)
        p_dis  = pulp.LpVariable.dicts("p_dis", P, 0, self.max_discharge_kw)
        soc    = pulp.LpVariable.dicts("soc", T, self.soc_min_kwh, self.soc_max_kwh)

        p_sell = pulp.LpVariable.dicts("p_sell", P, 0)
        p_buy  = pulp.LpVariable.dicts("p_buy", P, 0)

        y = pulp.LpVariable.dicts("y", P, 0, 1, cat="Binary")   # Lade-/Entlade-Exklusivität

        # Anfangs- und End-SOC festsetzen
        model += soc[0] == soc0
        model += soc[T[-1]] == soc0

        # Füge SOC Nebenbedingungen hinzu
        for t in range(1, len(T)):
            model += soc[t] == soc[t-1] + \
                delta_t * (self.eta_charge * p_ch[t-1] - p_dis[t-1] / self.eta_discharge)

        # Lade-/Entlade-Exklusivität
        for p in P:
            model += p_ch[p]  <= self.max_charge_kw * y[p]
            model += p_dis[p] <= self.max_discharge_kw * (1 - y[p])

        # Leistungsbilanz
        for p in P:
            model += (
                pv.iloc[p] + p_buy[p] + p_dis[p]
                ==
                load.iloc[p] + p_sell[p] + p_ch[p]
            )

        # Zielfunktion: Erlös – Kosten
        model += pulp.lpSum(
            (price_sell.iloc[p] * p_sell[p]
            - price_buy.iloc[p] * p_buy[p]) * delta_t
            for p in P
        )

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if verbose:
            print("Pulp-Resultate:")
            print(f"- Status numerisch: {model.status}")
            print(f"- Status verbal: {pulp.LpStatus[model.status]}")
            print(f"- optimaler Wert: {pulp.value(model.objective)}")

        ret_variables = {
            "soc":    pd.Series([soc[t].value() for t in T], index=time_points),
            "p_ch":   pd.Series([p_ch[p].value() for p in P], index=time_periods),
            "p_dis":  pd.Series([p_dis[p].value() for p in P], index=time_periods),
            "p_sell": pd.Series([p_sell[p].value() for p in P], index=time_periods),
            "p_buy":  pd.Series([p_buy[p].value() for p in P], index=time_periods),
        }

        return ret_variables

    def run(
        self,
        act_day: pd.Timestamp,
        verbose: bool = False,
        ) -> None:

        # VKW dynmaische Preise in ct/kWh
        self.df_prices_epex = pd.read_csv("../data/day_ahead_prices.csv", index_col=0)
        self.df_prices_epex.index = pd.to_datetime(self.df_prices_epex.index, utc=True)
        self.df_prices_epex.index = self.df_prices_epex.index.tz_convert("Europe/Vienna")
        prices_epex = self.df_prices_epex["day_ahead_price_EUR_MWh"]

        # get freq of act day
        day_data = prices_epex.loc[act_day: act_day + pd.Timedelta(days=1)]
        freq = pd.infer_freq(day_data.index)

        act_range = pd.date_range(
            start=act_day,
            end=act_day + pd.Timedelta(days=1),
            freq=freq,
            tz="Europe/Vienna",
            )
        self.prices_epex = self.df_prices_epex.loc[act_range]
        self.prices_epex = self.prices_epex / 1000  # Umrechnung in EUR/kWh
        price_sell = self.prices_epex - 0.6
        price_buy  = self.prices_epex + 1.44

        df_energy = pd.read_csv("../data/energy_data.csv", index_col=0)
        df_energy.index = pd.to_datetime(df_energy.index, utc=True)
        df_energy.index = df_energy.index.tz_convert("Europe/Vienna")
        df_energy = df_energy / 1000.0  # Umrechnung in kW

        df_energy = df_energy.resample(freq).mean()
        pv_forecast = df_energy["Production"].loc[act_range]
        load_forecast = df_energy["Consumption"].loc[act_range]
        self.net_load = load_forecast - pv_forecast

        self.lp_result = self.optimize_day(
            price_sell=price_sell,
            price_buy=price_buy,
            pv=pv_forecast,
            load=load_forecast,
            soc0=0.5 * self.capacity_kwh,
            verbose=verbose,
        )

    def plot_results(self):

        act_day = pd.Timestamp("2025-05-15", tz="Europe/Vienna")
        self.run(act_day)

        fig, axes = plt.subplots(
            nrows=5,
            ncols=1,
            figsize=(12, 8),
            sharex=True,
            )

        # --- Subplot 0: Day-Ahead Preis ---
        axes[0].step(
            self.prices_epex.index,
            self.prices_epex.values,
            where="post",
        )
        axes[0].set_ylabel("Preis [EUR/kWh]")
        axes[0].set_title("EPEX Day-Ahead Preis")
        axes[0].grid(True)

        # --- Subplot 1: Residuallast Profile ---
        axes[1].step(
            self.net_load.index,
            self.net_load.values,
            where="post",
        )
        axes[1].set_ylabel("Residuallast [kW]")
        axes[1].set_title("Residuallastprofil")
        axes[1].grid(True)

        # --- Subplot 2: SOC in Prozent ---
        soc_percent = 100 * self.lp_result["soc"] / self.capacity_kwh

        axes[2].step(
            soc_percent.index,
            soc_percent.values,
            where="post",
        )
        axes[2].set_ylabel("SOC [%]")
        axes[2].set_ylim(0, 100)
        axes[2].set_title("Batterie State of Charge")
        axes[2].grid(True)

        # --- Subplot 3: Einspeisung & Bezug ---
        axes[3].step(
            self.lp_result["p_sell"].index,
            self.lp_result["p_sell"].values,
            where="post",
            label="Einspeisung",
        )
        axes[3].step(
            self.lp_result["p_buy"].index,
            self.lp_result["p_buy"].values,
            where="post",
            label="Bezug",
        )
        axes[3].set_ylabel("Leistung [kW]")
        axes[3].set_title("Netz Einspeisung/Bezug")
        axes[3].grid(True)
        axes[3].legend()
        # --- Subplot 4: Ladeleistungen ---
        axes[4].step(
            self.lp_result["p_ch"].index,
            self.lp_result["p_ch"].values,
            where="post",
            label="Laden",
        )
        axes[4].step(
            self.lp_result["p_dis"].index,
            self.lp_result["p_dis"].values,
            where="post",
            label="Entladen",
        )

        axes[4].set_ylabel("Batterie Laden [kW]")
        axes[4].set_title("Batterieladung")
        axes[4].grid(True)
        axes[4].legend()

        # --- X-Achse ---
        locator = mdates.HourLocator(interval=2)      # alle 2 Stunden
        formatter = mdates.DateFormatter("%H:%M", tz="Europe/Vienna")     # nur Uhrzeit
        axes[4].xaxis.set_major_locator(locator)
        axes[4].xaxis.set_major_formatter(formatter)
        axes[4].set_xlim(self.prices_epex.index[0], self.prices_epex.index[-1])
        plt.setp(axes[4].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()
