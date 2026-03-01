from pathlib import Path
import time
import pandas as pd

from interfaces.mqtt import Victron_Mqtt_Reader
from interfaces.get_day_ahead_prices import DayAheadPrice
from interfaces.get_weather_data import WeatherDataRetriever
from forecasting.forecasting import ForecastingModel
from control.config import MpcConfig
from control.optimize import BessOptimizer

class MpcController:

    def run(self, my_config: MpcConfig = MpcConfig()):

        # Prepare main loop
        next_exec_time = time.monotonic()
        victron_mqtt_reader = Victron_Mqtt_Reader()
        my_forecaster = ForecastingModel()
        my_optimizer = BessOptimizer()

        while True:
            try:
                current_time = time.monotonic()

                # --- Run every 15 minutes ---
                if current_time >= next_exec_time:

                    next_exec_time += my_config.mpc_interval_sec

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
                        verbose=True,
                        )

                    set_netload_kw = (
                        netload_forecast_kw +
                        optimization_results["p_ch_kw"]
                        - optimization_results["p_dis_kw"]
                        )

                    victron_mqtt_reader.set_netload(netload_kw=set_netload_kw.iloc[0])

                    # Save results for analysis
                    results_df = pd.DataFrame({
                        "mpc_time": price_sell_eur_kwh.index,
                        "netload_forecast_kw": netload_forecast_kw,
                        "set_netload_kw": set_netload_kw,
                        "p_ch_kw": optimization_results["p_ch_kw"],
                        "p_dis_kw": optimization_results["p_dis_kw"],
                        "price_sell_eur_kwh": price_sell_eur_kwh,
                        "price_buy_eur_kwh": price_buy_eur_kwh,
                    })
                    file_path = Path(__file__).parent / "mpc_results.parquet"
                    results_df.reset_index(inplace=True)
                    if file_path.exists():
                        existing_df = pd.read_parquet(file_path)
                        results_df = pd.concat([existing_df, results_df], ignore_index=True)
                    results_df.to_parquet(file_path)

                    print("--- MPC Controller iteration completed. Next update in 15 minutes. ---")

                time.sleep(1)  # prevent CPU overload

            except KeyboardInterrupt:
                print("MPC Controller stopped.")
                break

            except Exception as e:
                print(f"Error in MPC Controller. Wait and retry. Error: {e}")
                time.sleep(20)

if __name__ == "__main__":
    mpc_controller = MpcController()
    mpc_controller.run()
