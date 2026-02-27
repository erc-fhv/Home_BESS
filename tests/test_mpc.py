from pathlib import Path
import pandas as pd

from interfaces.mqtt import Victron_Mqtt_Reader
from interfaces.get_day_ahead_prices import DayAheadPrice
from interfaces.get_weather_data import WeatherDataRetriever
from forecasting.forecasting import ForecastingModel
from control.optimize import BessOptimizer
from control.config import MpcConfig


def test_first_main_loop_run():

    # Prepare main loop
    victron_mqtt_reader = Victron_Mqtt_Reader()
    my_forecaster = ForecastingModel()
    my_optimizer = BessOptimizer()

    # Run and test one iteration of the main loop
    #
    act_soc = victron_mqtt_reader.get_latest_value("soc_percent")
    assert isinstance(act_soc, (int, float)), f"Expected act_soc to be a number, got {type(act_soc)}"
    assert 0 <= act_soc <= 100, f"Expected act_soc to be between 0 and 100, got {act_soc}"

    store_to_file = Path(__file__).parent / "test_day_ahead_prices.csv"
    price_sell_eur_kwh, price_buy_eur_kwh = DayAheadPrice.get_prices("vkw_dyn", store_to_file=store_to_file)
    assert price_sell_eur_kwh.index.equals(price_buy_eur_kwh.index), f"Got {price_sell_eur_kwh.index} and {price_buy_eur_kwh.index}"
    now = pd.Timestamp.now(tz=price_sell_eur_kwh.index.tz)
    assert (price_sell_eur_kwh.index >= now).all(), f"Got {price_sell_eur_kwh.index}"
    assert (price_sell_eur_kwh.index <= now + pd.Timedelta(hours=36)).all(), f"Got {price_sell_eur_kwh.index}"
    median_freq = price_sell_eur_kwh.index.to_series().diff().median()
    assert median_freq == pd.Timedelta(minutes=15), f"Got {median_freq}"
    assert len(price_sell_eur_kwh) <= 4*36, f"Got {len(price_sell_eur_kwh)}"
    assert len(price_sell_eur_kwh) >= 4*8, f"Got {len(price_sell_eur_kwh)}"
    assert price_sell_eur_kwh.min() >= -0.2, f"Got {price_sell_eur_kwh.min()}"
    assert price_sell_eur_kwh.max() <= 0.8, f"Got {price_sell_eur_kwh.max()}"
    assert price_buy_eur_kwh.min() >= -0.2, f"Got {price_buy_eur_kwh.min()}"
    assert price_buy_eur_kwh.max() <= 0.8, f"Got {price_buy_eur_kwh.max()}"
    assert price_buy_eur_kwh.std() >= 0.01, f"Got {price_buy_eur_kwh.std()}"    # 1cent std

    weather_data = WeatherDataRetriever.retrieve_weather_data(time_range=price_sell_eur_kwh.index)
    assert weather_data.index.equals(price_buy_eur_kwh.index), f"Got {weather_data.index} and {price_buy_eur_kwh.index}"
    assert (weather_data.index >= now).all(), f"Got {weather_data.index}"
    assert (weather_data.index <= now + pd.Timedelta(hours=36)).all(), f"Got {weather_data.index}"
    median_freq = weather_data.index.to_series().diff().median()
    assert median_freq == pd.Timedelta(minutes=15), f"Got {median_freq}"
    assert len(weather_data) <= 4*36, f"Got {len(weather_data)}"
    assert len(weather_data) >= 4*8, f"Got {len(weather_data)}"
    assert (weather_data["temperature_2m"] >= -30).all(), f"Got {weather_data['temperature_2m'].min()}"
    assert (weather_data["temperature_2m"] <= 50).all(), f"Got {weather_data['temperature_2m'].max()}"
    assert weather_data["temperature_2m"].std() >= 1, f"Got {weather_data['temperature_2m'].std()}"

    netload_forecast_kw = my_forecaster.predict(weather_data)
    assert netload_forecast_kw.index.equals(price_buy_eur_kwh.index), f"Got {netload_forecast_kw.index} and {price_buy_eur_kwh.index}"
    assert (netload_forecast_kw.index >= now).all(), f"Got {netload_forecast_kw.index}"
    assert (netload_forecast_kw.index <= now + pd.Timedelta(hours=36)).all(), f"Got {netload_forecast_kw.index}"
    median_freq = netload_forecast_kw.index.to_series().diff().median()
    assert median_freq == pd.Timedelta(minutes=15), f"Got {median_freq}"
    assert len(netload_forecast_kw) <= 4*36, f"Got {len(netload_forecast_kw)}"
    assert len(netload_forecast_kw) >= 4*8, f"Got {len(netload_forecast_kw)}"
    assert netload_forecast_kw.min() >= -15, f"Got {netload_forecast_kw.min()}"
    assert netload_forecast_kw.max() <= 20, f"Got {netload_forecast_kw.max()}"
    assert netload_forecast_kw.std() >= 1, f"Got {netload_forecast_kw.std()}"

    optimization_results = my_optimizer.optimize(
        my_config=MpcConfig(),
        price_sell_eur_kwh=price_sell_eur_kwh,
        price_buy_eur_kwh=price_buy_eur_kwh,
        net_load_kw=netload_forecast_kw,
        soc_init_percent=act_soc,
        verbose=True,
        )
    assert optimization_results["p_ch_kw"].index.equals(optimization_results["p_dis_kw"].index), f"Got {optimization_results['p_ch_kw'].index} and {optimization_results['p_dis_kw'].index}"
    assert (optimization_results["p_ch_kw"].index >= now).all(), f"Got {optimization_results['p_ch_kw'].index}"
    assert (optimization_results["p_ch_kw"].index <= now + pd.Timedelta(hours=36)).all(), f"Got {optimization_results['p_ch_kw'].index}"
    median_freq = optimization_results["p_ch_kw"].index.to_series().diff().median()
    assert median_freq == pd.Timedelta(minutes=15), f"Got {median_freq}"
    assert len(optimization_results["p_ch_kw"]) <= 4*36, f"Got {len(optimization_results['p_ch_kw'])}"
    assert len(optimization_results["p_ch_kw"]) >= 4*8, f"Got {len(optimization_results['p_ch_kw'])}"
    assert optimization_results["p_ch_kw"].min() >= -12, f"Got {optimization_results['p_ch_kw'].min()}"
    assert optimization_results["p_ch_kw"].max() <= 12, f"Got {optimization_results['p_ch_kw'].max()}"
    assert optimization_results["p_ch_kw"].std() >= 0.1, f"Got {optimization_results['p_ch_kw'].std()}"

    set_netload_kw = (
        netload_forecast_kw.iloc[0] +
        optimization_results["p_ch_kw"].iloc[0]
        - optimization_results["p_dis_kw"].iloc[0]
        )
    print("molu: set_netload_kw", set_netload_kw)
    assert isinstance(set_netload_kw, (int, float)), f"Expected set_netload_kw to be a number, got {type(set_netload_kw)}"
    assert set_netload_kw >= -12, f"Got {set_netload_kw}"
    assert set_netload_kw <= 12, f"Got {set_netload_kw}"
