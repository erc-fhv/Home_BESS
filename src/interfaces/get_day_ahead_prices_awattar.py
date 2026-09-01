from pathlib import Path
import requests
import pandas as pd

class AwattarPrice:
    """Class to read out day-ahead electricity prices from the aWATTar API."""

    BASE_URLS = {
        "AT": "https://api.awattar.at/v1/marketdata",
        "DE": "https://api.awattar.de/v1/marketdata",
    }

    @staticmethod
    def get_epex_prices(
        country_code="AT",
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        store_to_file: Path | None = None,
        ) -> pd.Series:
        """Return day-ahead Epex electricity prices in EUR/kWh."""

        if country_code not in AwattarPrice.BASE_URLS:
            raise ValueError(f"Unsupported country_code for aWATTar: {country_code}")

        if start_date is None:
            # Take the current time rounded down to the nearest 15 minutes
            start_date = pd.Timestamp.now(tz="Europe/Vienna").floor("15min")
        if end_date is None:
            # Price horizon is max. 1.5 days, so 2 days ensures we get all relevant prices
            end_date = start_date + pd.Timedelta(days=2)

        # Readout day-ahead prices (aWATTar's fair-use limit is 100 requests/day)
        params = {
            "start": int(start_date.tz_convert("UTC").timestamp() * 1000),
            "end": int(end_date.tz_convert("UTC").timestamp() * 1000),
        }
        response = requests.get(AwattarPrice.BASE_URLS[country_code], params=params, timeout=10)
        response.raise_for_status()
        entries = response.json()["data"]

        index = pd.DatetimeIndex(
            pd.to_datetime([entry["start_timestamp"] for entry in entries], unit="ms", utc=True)
        ).tz_convert("Europe/Vienna")
        prices = pd.Series([entry["marketprice"] for entry in entries], index=index).sort_index()
        prices = prices[(prices.index >= start_date) & (prices.index < end_date)]

        # Check the returned data
        assert isinstance(prices.index, pd.DatetimeIndex), \
            f"Expected DatetimeIndex, got {type(prices.index)}"
        assert prices.index.tz is not None and str(prices.index.tz) == "Europe/Vienna", \
            f"Expected timezone 'Europe/Vienna', got {prices.index.tz}"
        if len(prices) > 1:
            median_freq = prices.index.to_series().diff().median()
            assert median_freq == pd.Timedelta(minutes=15) \
                or median_freq == pd.Timedelta(minutes=60), \
                f"Expected frequency of 15 or 60 minutes, got {median_freq}"

        # Convert prices from EUR/MWh to EUR/kWh
        prices = prices / 1000.0

        # Save to CSV if file path is provided
        if store_to_file is not None:
            prices.index.name = "timestamp"
            prices.name = "day_ahead_price_eur_kWh"
            prices.to_csv(store_to_file)

        return prices
