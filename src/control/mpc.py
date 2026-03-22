from pathlib import Path
import time
import tomllib
import pandas as pd
import numpy as np

from interfaces.mqtt import Victron_Mqtt_Reader
from interfaces.get_day_ahead_prices import DayAheadPrice
from interfaces.get_weather_data import WeatherDataRetriever
from forecasting.forecasting import ForecastingModel
from control.optimize import BessOptimizer

class MpcController:

    def __init__(self):
        self.output_file_timestamp = None

    def run(self):

        # Prepare main loop
        next_exec_time_15min = pd.Timestamp.now(tz="UTC")
        fast_period = 120  # seconds
        victron_mqtt_reader = Victron_Mqtt_Reader()
        my_forecaster = ForecastingModel()
        my_optimizer = BessOptimizer()
        self.output_file_timestamp = pd.Timestamp.now(tz="UTC")
        set_netload_kw = None
        next_fast_cycle = next_exec_time_15min + pd.Timedelta(seconds=fast_period)

        while True:
            try:
                current_time = pd.Timestamp.now(tz="UTC")

                # --- Run every 15 minutes ---
                if current_time >= next_exec_time_15min:

                    mpc_time = current_time

                    my_config = self.load_config()

                    act_soc_percent = victron_mqtt_reader.get_latest_value("soc_percent")

                    price_sell_eur_kwh, price_buy_eur_kwh = DayAheadPrice.get_prices("vkw_dyn")

                    assert isinstance(price_sell_eur_kwh.index, pd.DatetimeIndex)

                    weather_data = WeatherDataRetriever.retrieve_weather_data(
                        time_range = price_sell_eur_kwh.index)

                    netload_forecast_kw = my_forecaster.predict(weather_data)

                    optimization_results = my_optimizer.optimize(
                        my_config=my_config,
                        price_sell_eur_kwh=price_sell_eur_kwh,
                        price_buy_eur_kwh=price_buy_eur_kwh,
                        net_load_kw=netload_forecast_kw,
                        soc_init_percent=act_soc_percent,
                        verbose=False,
                        )

                    set_netload_kw = optimization_results["set_netload_kw"].iloc[0]
                    victron_mqtt_reader.set_netload(netload_kw=set_netload_kw)

                    self.save_results(price_sell_eur_kwh, price_buy_eur_kwh, netload_forecast_kw,
                        optimization_results, mpc_time, weather_data)

                    victron_mqtt_reader.start_heartbeat()

                    next_exec_time_15min = current_time.floor("15min") + pd.Timedelta(minutes=15)

                    next_fast_cycle = current_time + pd.Timedelta(seconds=fast_period)

                # --- Run every fast_period seconds ---
                elif current_time >= next_fast_cycle:
                    try:
                        act_netload_w = victron_mqtt_reader.get_latest_value("netload_read")
                        act_netload_kw = act_netload_w / 1000.0
                        if not np.isclose(act_netload_kw, set_netload_kw, rtol=1e-2):
                            print(f"Warning: Set net load {set_netload_kw:.2f} kW does not match " + \
                                f"actual net load {act_netload_kw:.2f} kW.")
                        next_fast_cycle += pd.Timedelta(seconds=fast_period)
                    except Exception as e:
                        print(f"Error reading actual net load. Wait and retry. Error: {e}")

                else:
                    time.sleep(1)  # prevent CPU overload

            except Exception as e:
                print(f"Error in MPC Controller. Wait and retry. Error: {e}")
                victron_mqtt_reader.stop_heartbeat()
                time.sleep(20)

    def save_results(
        self,
        price_sell_eur_kwh,
        price_buy_eur_kwh,
        netload_forecast_kw,
        optimization_results,
        mpc_time,
        weather_data,
        ) -> None:
        """Save MPC results to a Parquet file for later analysis."""

        results_df = pd.DataFrame({
            "mpc_time": mpc_time,
            "netload_forecast_kw": netload_forecast_kw,
            "price_sell_eur_kwh": price_sell_eur_kwh,
            "price_buy_eur_kwh": price_buy_eur_kwh,
        })
        results_df = pd.concat([results_df, pd.DataFrame(optimization_results)], axis=1)
        results_df = pd.concat([results_df, pd.DataFrame(weather_data)], axis=1)

        results_df = results_df.reset_index(names="timestamp")
        timestamp = self.output_file_timestamp.strftime('%Y%m%d_%H%M%S')
        file_path = Path(__file__).parent / f"output/mpc_results_{timestamp}.parquet"
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)
        results_df.to_parquet(file_path, engine="pyarrow")

    def load_config(self) -> dict:
        """Load MPC configuration from a TOML file."""

        config_file = Path(__file__).parent / "config.toml"
        with open(config_file, "rb") as f:
            return tomllib.load(f)

if __name__ == "__main__":
    mpc_controller = MpcController()
    mpc_controller.run()
