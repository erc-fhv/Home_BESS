import time

from home_bess.interfaces.get_day_ahead_prices import DayAheadPrice
from home_bess.interfaces.get_weather_data import WeatherDataRetriever
from home_bess.interfaces.mqtt import Victron_Mqtt_Reader


class MpcController:
    def __init__(self):
        self.victron_mqtt_reader = None

    def run(self):

        # Setup next execution times
        update_interval = 60 * 15  # 15 minutes
        next_exec_time = time.monotonic() + update_interval

        self.victron_mqtt_reader = Victron_Mqtt_Reader()

        while True:
            try:
                current_time = time.monotonic()

                # --- Run every 15 minutes ---
                if current_time < next_exec_time:
                    next_exec_time += update_interval

                    self.victron_mqtt_reader.get_latest_value("soc")
                    DayAheadPrice.get_prices()
                    WeatherDataRetriever.retrieve_weather_data()

                    return

            except KeyboardInterrupt:
                print("MPC Controller stopped.")
                break


if __name__ == "__main__":
    mpc_controller = MpcController()
    mpc_controller.run()
