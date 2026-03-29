from pathlib import Path
from typing import Callable
import pulp
import pandas as pd

from interfaces.get_day_ahead_prices import DayAheadPrice
from control.optimize import BessOptimizer

class Bess:
    def __init__(self) -> None:

        self.capacity_kwh = None
        self.max_charge_kw = None
        self.max_discharge_kw = None
        self.soc_min_percent = None
        self.soc_final_percent = None
        self.eta_charge = None
        self.eta_discharge = None
        self.net_load_of_one_day_kw = None
        self.act_prices_epex_eur_kwh = None
        self.price_sell_eur_kwh = pd.Series()
        self.price_buy_eur_kwh = pd.Series()

        self._load_epex_prices()

        file_path = Path(__file__).parent / "data" / "example_household_without_battery.csv"
        self.netload_kw = pd.read_csv(file_path, index_col=0)
        self.netload_kw.index = pd.to_datetime(self.netload_kw.index, utc=True)
        self.netload_kw.index = self.netload_kw.index.tz_convert("Europe/Vienna")

    def set_netload_profile(self, netload_kw: pd.DataFrame) -> None:
        """Set the netload profile for the simulation."""
        assert isinstance(netload_kw.index, pd.DatetimeIndex), "index must be datetime index."
        assert netload_kw.index.tz is not None, "Netload index must be timezone-aware."
        assert netload_kw.index.tz.zone == "Europe/Vienna", "Netload index must be 'Vienna' tz."
        self.netload_kw = netload_kw

    def get_netload_profile(self) -> pd.DataFrame:
        """Returns the currently set netload profile."""
        return self.netload_kw

    def run(
        self,
        act_day: pd.Timestamp,
        use_dynamic_prices: bool,
        epex_offset_buy_eur_kwh: float,
        epex_offset_sell_eur_kwh: float,
        grid_fee_eur_kwh: float,
        vat: float,
        fix_price_buy_eur_kwh: float,
        fix_price_sell_eur_kwh: float,
        verbose: bool = False,
        control_algorithm: str = "model-predictive-control",
        soc_init_percent: float = 50.0,
        ) -> dict[str, pd.Series]:
        """Run the BESS optimization for a given day and return the results as a dictionary."""

        # Treat zero capacity as "no battery"
        effective_algorithm = control_algorithm
        if control_algorithm != "no-control" and not self.capacity_kwh:
            effective_algorithm = "no-control"

        if effective_algorithm != "no-control":
            assert self.capacity_kwh is not None, "Battery capacity must be set."
            assert self.max_charge_kw is not None, "Max charge power must be set."
            assert self.max_discharge_kw is not None, "Max discharge power must be set."
            assert self.soc_min_percent is not None, "Minimum SOC must be set."
            assert self.soc_final_percent is not None, "Maximum SOC must be set."
            assert self.eta_charge is not None, "Charging efficiency must be set."
            assert self.eta_discharge is not None, "Discharging efficiency must be set."

        act_range = pd.date_range(
            start=act_day,
            end=act_day + pd.Timedelta(days=1),
            freq='15min',
            tz="Europe/Vienna",
            inclusive="left",
            )
        self.act_prices_epex_eur_kwh = self.prices_epex_eur_kWh.loc[act_range]

        if use_dynamic_prices:
            self.price_sell_eur_kwh = self.act_prices_epex_eur_kwh - epex_offset_sell_eur_kwh
            self.price_buy_eur_kwh  = (self.act_prices_epex_eur_kwh + epex_offset_buy_eur_kwh \
                + grid_fee_eur_kwh) * (1 + vat)
        else:
            self.price_sell_eur_kwh = pd.Series(fix_price_sell_eur_kwh,
                index=self.act_prices_epex_eur_kwh.index)
            self.price_buy_eur_kwh  = pd.Series((fix_price_buy_eur_kwh + grid_fee_eur_kwh) \
                * (1 + vat), index=self.act_prices_epex_eur_kwh.index)

        self.net_load_of_one_day_kw = self.netload_kw.loc[act_range]["net_load_kw"]

        my_optimizer = BessOptimizer()

        if effective_algorithm == "no-control":
            lp_results = my_optimizer.no_optimize(
                price_sell_eur_kwh=self.price_sell_eur_kwh,
                price_buy_eur_kwh=self.price_buy_eur_kwh,
                net_load_kw=self.net_load_of_one_day_kw,
                verbose=verbose,
            )
        elif effective_algorithm == "pv-ueberschussladen":
            lp_results = my_optimizer.pv_surplus_charge(
                price_sell_eur_kwh=self.price_sell_eur_kwh,
                price_buy_eur_kwh=self.price_buy_eur_kwh,
                net_load_kw=self.net_load_of_one_day_kw,
                soc_init_percent=soc_init_percent,
                capacity_kwh=self.capacity_kwh,
                max_charge_kw=self.max_charge_kw,
                max_discharge_kw=self.max_discharge_kw,
                soc_min_percent=self.soc_min_percent,
                soc_max_percent=100.0,
                eta_charge=self.eta_charge,
                eta_discharge=self.eta_discharge,
                verbose=verbose,
            )
        else:
            lp_results = my_optimizer.optimize_milp(
                price_sell_eur_kwh=self.price_sell_eur_kwh,
                price_buy_eur_kwh=self.price_buy_eur_kwh,
                net_load_kw=self.net_load_of_one_day_kw,
                soc_init_percent=soc_init_percent,
                soc_final_percent=soc_init_percent,
                capacity_kwh=self.capacity_kwh,
                max_charge_kw=self.max_charge_kw,
                max_discharge_kw=self.max_discharge_kw,
                soc_min_percent=self.soc_min_percent,
                eta_charge=self.eta_charge,
                eta_discharge=self.eta_discharge,
                soc_max_percent=self.soc_final_percent,
                verbose=verbose,
            )

        lp_results["act_prices_epex_eur_kwh"] = self.act_prices_epex_eur_kwh
        lp_results["price_buy_eur_kwh"] = self.price_buy_eur_kwh
        lp_results["price_sell_eur_kwh"] = self.price_sell_eur_kwh
        lp_results["net_load_kw"] = self.net_load_of_one_day_kw

        return lp_results

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
        control_algorithm: str = "model-predictive-control",
    ) -> pd.DataFrame:
        """Führt die Simulation für jeden Tag in [start_day, end_day] aus."""

        start_ts = start_day.normalize()
        end_ts = end_day.normalize()
        if end_ts < start_ts:
            raise ValueError("end_day must be greater than or equal to start_day")
        all_days = pd.date_range(start=start_ts, end=end_ts, freq="1D", tz="Europe/Vienna")
        rows = []
        soc_init = self.soc_final_percent if self.soc_final_percent is not None else 50.0

        for idx, act_day in enumerate(all_days):

            lp_results = self.run(
                act_day=act_day,
                use_dynamic_prices=use_dynamic_prices,
                epex_offset_buy_eur_kwh=epex_offset_buy,
                epex_offset_sell_eur_kwh=epex_offset_sell,
                grid_fee_eur_kwh=grid_fee,
                vat=vat,
                fix_price_buy_eur_kwh=fix_price_buy,
                fix_price_sell_eur_kwh=fix_price_sell,
                verbose=verbose,
                control_algorithm=control_algorithm,
                soc_init_percent=soc_init,
                )
            rows.append(lp_results)

            # SOC carry-over: use end-of-day SOC as start for next day
            soc_series = lp_results.get("soc_percent")
            if isinstance(soc_series, pd.Series) and len(soc_series) > 0:
                soc_init = float(soc_series.iloc[-1])

            if progress_callback is not None:
                progress_callback(idx+1, len(all_days), act_day, lp_results)

        lp_results_df = pd.DataFrame(rows)
        assert not lp_results_df.empty, "Simulation returned no results."
        lp_results_df["date"] = pd.to_datetime(lp_results_df["date"])
        lp_results_df = lp_results_df.set_index("date").sort_index()

        return lp_results_df

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
            self.soc_min_percent = soc_min_percent
        if soc_final_percent is not None:
            self.soc_final_percent = soc_final_percent
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

        self.prices_epex_eur_kWh = df_prices["day_ahead_price_eur_kWh"]
        self.prices_epex_eur_kWh = self.prices_epex_eur_kWh.resample('15min').ffill()

if __name__ == "__main__":
    from simulation.web_app import run_dashboard
    bess = Bess()
    run_dashboard(bess)
