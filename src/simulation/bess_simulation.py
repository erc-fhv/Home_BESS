from pathlib import Path
import pulp
import numpy as np
import pandas as pd

from interfaces.get_day_ahead_prices import DayAheadPrice


class Bess:
    def __init__(
        self,
        capacity_kwh: float = 5.12*6,
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
        self.price_sell = pd.Series()
        self.price_buy = pd.Series()

        self._load_epex_prices()

        # Energieverbrauchs- und Produktionsdaten einlesen
        # todo: ebenfalls automatisch einlesen, falls nicht vorhanden.
        file_path = Path(__file__).parent / "data" / "example_household_without_battery.csv"
        self.df_energy = pd.read_csv(file_path, index_col=0)
        self.df_energy.index = pd.to_datetime(self.df_energy.index, utc=True)
        self.df_energy.index = self.df_energy.index.tz_convert("Europe/Vienna")

    def _load_epex_prices(self) -> None:
        """EPEX-Preise aus CSV laden und fehlende Daten via API nachladen."""
        file_path = Path(__file__).parent / "data" / "epex_prices.csv"
        now = pd.Timestamp.now(tz="Europe/Vienna").floor("1D")

        if file_path.exists():
            df_prices = pd.read_csv(file_path, index_col=0)
            df_prices.index = pd.to_datetime(df_prices.index, utc=True)
            df_prices.index = df_prices.index.tz_convert("Europe/Vienna")
            last_ts = df_prices.index.max().floor("1D")

            if last_ts < now:
                new_prices = DayAheadPrice.get_epex_prices(
                    country_code="AT",
                    start_date=last_ts,
                    end_date=now,
                )
                if not new_prices.empty:
                    df_new = new_prices.to_frame(name="day_ahead_price_eur_kWh")
                    df_prices = pd.concat([df_prices, df_new])
                    df_prices = df_prices[
                        ~df_prices.index.duplicated(keep="last")]
                    df_prices.sort_index(inplace=True)
                    df_prices.index.name = "timestamp"
                    df_prices.to_csv(file_path)
        else:
            raise FileNotFoundError(f"EPEX price file '{file_path}' not found.")

        self.prices_epex = df_prices["day_ahead_price_eur_kWh"]
        self.prices_epex = self.prices_epex.resample('15min').ffill()

    def optimize_day(
        self,
        price_sell: pd.Series,    # ct/kWh
        price_buy: pd.Series,     # ct/kWh
        net_load: pd.Series,      # kW
        soc0: float,              # kWh
        verbose: bool = True,
        ) -> tuple:

        # Konsistenzcheck
        assert price_sell.index.equals(price_buy.index)
        assert price_buy.index.equals(net_load.index)

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
                p_buy[p] + p_dis[p] - net_load.iloc[p] - p_sell[p] - p_ch[p]
                ==
                0.0
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

        return ret_variables, model

    def run(
        self,
        act_day: pd.Timestamp,
        use_dynamic_prices: bool = True,
        verbose: bool = False,
        ) -> None:

        act_range = pd.date_range(
            start=act_day,
            end=act_day + pd.Timedelta(days=1),
            freq='15min',
            tz="Europe/Vienna",
            )
        self.act_prices_epex = self.prices_epex.loc[act_range]

        if use_dynamic_prices:
            # VKW dynmaische Preise in EUR/kWh
            self.price_sell = self.act_prices_epex - 0.006
            self.price_buy  = self.act_prices_epex + 0.0144
        else:
            # fixe Preise in ct/kWh
            self.price_sell = pd.Series(0.09, index=self.act_prices_epex.index)
            self.price_buy  = pd.Series(0.1272, index=self.act_prices_epex.index)

        self.net_load_kw = self.df_energy.loc[act_range]["net_load_kw"]

        self.lp_result, self.pulp_model = self.optimize_day(
            price_sell=self.price_sell,
            price_buy=self.price_buy,
            net_load=self.net_load_kw,
            soc0=0.5 * self.capacity_kwh,
            verbose=verbose,
        )


if __name__ == "__main__":
    from simulation.visualization import run_dashboard
    bess = Bess()
    run_dashboard(bess)
