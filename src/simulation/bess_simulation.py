from pathlib import Path
from typing import Callable
import pulp
import pandas as pd

from interfaces.get_day_ahead_prices import DayAheadPrice


class Bess:
    def __init__(self) -> None:

        self.capacity_kwh = None
        self.max_charge_kw = None
        self.max_discharge_kw = None
        self.soc_min_kwh = None
        self.soc_max_kwh = None
        self.eta_charge = None
        self.eta_discharge = None
        self.lp_result = {}
        self.price_sell = pd.Series()
        self.price_buy = pd.Series()

        # Lade EPEX-Preise und Energieverbrauchsdaten
        self._load_epex_prices()

        # Energieverbrauchs- und Produktionsdaten einlesen
        file_path = Path(__file__).parent / "data" / "example_household_without_battery.csv"
        self.df_energy = pd.read_csv(file_path, index_col=0)
        self.df_energy.index = pd.to_datetime(self.df_energy.index, utc=True)
        self.df_energy.index = self.df_energy.index.tz_convert("Europe/Vienna")

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
        assert self.capacity_kwh is not None, "Battery capacity must be set."
        assert self.max_charge_kw is not None, "Max charge power must be set."
        assert self.max_discharge_kw is not None, "Max discharge power must be set."
        assert self.soc_min_kwh is not None, "Minimum SOC must be set."
        assert self.soc_max_kwh is not None, "Maximum SOC must be set."
        assert self.eta_charge is not None, "Charging efficiency must be set."
        assert self.eta_discharge is not None, "Discharging efficiency must be set."

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
        use_dynamic_prices: bool,
        epex_offset_buy: float,
        epex_offset_sell: float,
        grid_fee: float,
        vat: float,
        fix_price_buy: float,
        fix_price_sell: float,
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
            # VKW dynamische Preise in EUR/kWh
            self.price_sell = self.act_prices_epex - epex_offset_sell
            self.price_buy  = (self.act_prices_epex + epex_offset_buy + grid_fee) * (1 + vat)
        else:
            # fixe Preise in EUR/kWh
            self.price_sell = pd.Series(fix_price_sell, index=self.act_prices_epex.index)
            self.price_buy  = pd.Series((fix_price_buy + grid_fee) * (1 + vat), index=self.act_prices_epex.index)

        self.net_load_kw = self.df_energy.loc[act_range]["net_load_kw"]

        self.lp_result, self.pulp_model = self.optimize_day(
            price_sell=self.price_sell,
            price_buy=self.price_buy,
            net_load=self.net_load_kw,
            soc0=0.5 * self.capacity_kwh,
            verbose=verbose,
        )

    def run_total_simulation(
        self,
        start_day: pd.Timestamp,
        end_day: pd.Timestamp,
        use_dynamic_prices: bool,
        epex_offset_buy: float,
        epex_offset_sell: float,
        grid_fee: float,
        vat: float,
        fix_price_buy: float,
        fix_price_sell: float,
        verbose: bool = False,
        progress_callback: Callable | None = None,
    ) -> pd.DataFrame:
        """Führt die Simulation für jeden Tag in [start_day, end_day] aus."""

        start_ts = start_day.normalize()
        end_ts = end_day.normalize()
        if end_ts < start_ts:
            raise ValueError("end_day must be greater than or equal to start_day")
        all_days = pd.date_range(start=start_ts, end=end_ts, freq="1D", tz="Europe/Vienna")
        rows = []

        for idx, act_day in enumerate(all_days):

            self.run(
                act_day=act_day,
                use_dynamic_prices=use_dynamic_prices,
                epex_offset_buy=epex_offset_buy,
                epex_offset_sell=epex_offset_sell,
                grid_fee=grid_fee,
                vat=vat,
                fix_price_buy=fix_price_buy,
                fix_price_sell=fix_price_sell,
                verbose=verbose,
            )
            day_metrics = self.get_current_day_metrics(act_day=act_day)
            rows.append(day_metrics)

            if progress_callback is not None:
                progress_callback(idx+1, len(all_days), act_day, day_metrics)

        result_df = pd.DataFrame(rows)
        assert not result_df.empty, "Simulation returned no results."
        result_df["date"] = pd.to_datetime(result_df["date"])
        result_df = result_df.set_index("date").sort_index()

        return result_df

    def get_current_day_metrics(self, act_day: pd.Timestamp) -> dict:
        """Berechnet Tageskennzahlen aus den letzten Optimierungsergebnissen."""

        if self.net_load_kw.empty:
            raise ValueError("No simulation results available. Run `run()` first.")

        # Energie aus Leistung mit der im Datensatz enthaltenen Schrittweite berechnen.
        delta_t = (self.net_load_kw.index[1] - self.net_load_kw.index[0]).total_seconds() / 3600.0

        objective_value = float(pulp.value(self.pulp_model.objective) or 0.0)
        grid_import_kwh = float(self.lp_result["p_buy"].sum() * delta_t)
        grid_export_kwh = float(self.lp_result["p_sell"].sum() * delta_t)
        total_load_kwh = float(self.net_load_kw.clip(lower=0.0).sum() * delta_t)

        return {
            "date": pd.Timestamp(act_day).normalize().date(),
            "profit_eur": objective_value,
            "grid_import_kwh": grid_import_kwh,
            "grid_export_kwh": grid_export_kwh,
            "total_load_kwh": total_load_kwh,
        }

    def update_battery_params(
        self,
        capacity_kwh: float | None = None,
        max_charge_kw: float | None = None,
        max_discharge_kw: float | None = None,
        soc_min_percent: float | None = None,
        soc_final_percent: float | None = None,
        eta_charge: float | None = None,
        eta_discharge: float | None = None,
    ) -> None:
        """Aktualisiert die Batterie-Parameter."""

        if capacity_kwh is not None:
            self.capacity_kwh = capacity_kwh
        if max_charge_kw is not None:
            self.max_charge_kw = max_charge_kw
        if max_discharge_kw is not None:
            self.max_discharge_kw = max_discharge_kw
        if soc_min_percent is not None:
            self.soc_min_kwh = (soc_min_percent / 100.0) * self.capacity_kwh
        if soc_final_percent is not None:
            self.soc_max_kwh = (soc_final_percent / 100.0) * self.capacity_kwh
        if eta_charge is not None:
            self.eta_charge = eta_charge
        if eta_discharge is not None:
            self.eta_discharge = eta_discharge

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

if __name__ == "__main__":
    from simulation.visualization import run_dashboard
    bess = Bess()
    run_dashboard(bess)
