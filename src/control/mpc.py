from logging import config
import time

from interfaces.mqtt import Victron_Mqtt_Reader
from interfaces.get_day_ahead_prices import DayAheadPrice
from interfaces.get_weather_data import WeatherDataRetriever
from forecasting.forecasting import ForecastingModel
from control.config import MpcConfig
from control.optimize import BessOptimizer

class MpcController:
    def __init__(self, my_config: MpcConfig = MpcConfig()):
        self.victron_mqtt_reader = None
        self.config = my_config

    def run(self):

        # Prepare main loop
        next_exec_time = time.monotonic() + self.config.update_interval_sec
        self.victron_mqtt_reader = Victron_Mqtt_Reader()
        my_forecaster = ForecastingModel()
        my_optimizer = BessOptimizer()

        while True:
            try:
                current_time = time.monotonic()

                # --- Run every 15 minutes ---
                if current_time < next_exec_time:

                    next_exec_time += self.config.update_interval_sec

                    act_soc = self.victron_mqtt_reader.get_latest_value("soc")

                    price_sell, price_buy = DayAheadPrice.get_prices("vkw")

                    weather_data = WeatherDataRetriever.retrieve_weather_data()

                    netload_forecast = my_forecaster.predict(weather_data)

                    optimization_results = my_optimizer.optimize(
                        my_config=self.config,
                        price_sell_ct_kwh=price_sell,
                        price_buy_ct_kwh=price_buy,
                        net_load_kw=netload_forecast,
                        soc_init_percent=act_soc,
                        verbose=True,
                        )

                    set_netload = (
                        netload_forecast[0] +
                        optimization_results["p_ch"][0]
                        - optimization_results["p_dis"][0]
                        )

                    self.victron_mqtt_reader.set_netload(set_netload)

            except KeyboardInterrupt:
                print("MPC Controller stopped.")
                break

            except Exception as e:
                print(f"Error in MPC Controller. Wait and retry. Error: {e}")
                time.sleep(20)

if __name__ == "__main__":
    mpc_controller = MpcController()
    mpc_controller.run()
