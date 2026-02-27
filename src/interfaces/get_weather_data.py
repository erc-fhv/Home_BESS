"""Get weather measurements"""

from time import strftime
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


class WeatherDataRetriever:
    """Class to retrieve weather data from the Open-Meteo API and store it as a CSV file."""

    @staticmethod
    def retrieve_weather_data(
        time_range: pd.DatetimeIndex,
        latitude: float = 47.38517, # Bezau
        longitude: float = 9.895996, # Bezau
        timezone: str = "Europe/Vienna",
        weather_actuality: str = "future_forecast",
        ) -> pd.DataFrame:

        # Define the weather features to retrieve.
        weather_features = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
            "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
            "direct_radiation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
            "precipitation"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": time_range[0].strftime("%Y-%m-%d"),
            "end_date": time_range[-1].strftime("%Y-%m-%d"),
            "timezone": timezone,
            "minutely_15": weather_features,
        }

        if weather_actuality == "future_forecast":
            url = "https://api.open-meteo.com/v1/forecast"
        elif weather_actuality == "actual":
            # Note: The historical forecast API provides 15min for historical data.
            url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        else:
            raise ValueError(f"Invalid weather_actuality value: {weather_actuality}. "
                "Expected 'actual' or 'future_forecast'.")

        # Do the API request to retrieve the weather data.
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        responses = openmeteo.weather_api(url, params=params)

        # Create a DataFrame with the minutely_15 data.
        minutely_15 = responses[0].Minutely15()
        assert minutely_15 is not None, "Minutely_15 data is missing in the API response."
        data = {}
        for i, param in enumerate(weather_features):
            act_variables = minutely_15.Variables(i)
            assert act_variables is not None, f"Minutely_15 variable '{param}' is missing."
            data[param] = act_variables.ValuesAsNumpy()
        data["date"] = pd.date_range(
            start = pd.to_datetime(minutely_15.Time(), unit = "s", utc = True),
            end =  pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = minutely_15.Interval()),
            inclusive = "left")
        df = pd.DataFrame(data = data)
        df.set_index("date", inplace=True)

        # Convert to given timezone
        assert df.index.tz is not None, \
            "Minutely_15 DataFrame index must be timezone-aware."
        df = df.tz_convert(timezone)

        # Filter to the given date range
        df = df.loc[time_range[0]:time_range[-1]]

        return df
