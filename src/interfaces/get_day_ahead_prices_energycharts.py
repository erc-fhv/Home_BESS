from pathlib import Path
import requests
import pandas as pd

class EnergyChartsPrice:
    """Class to read out day-ahead electricity prices from the Energy-Charts API
    (Fraunhofer ISE), which publishes true 15-minute EPEX day-ahead prices."""

    BASE_URL = "https://api.energy-charts.info/v2/price"

    BIDDING_ZONES = {
        "AT": "AT",
        "DE": "DE-LU",
    }

    @staticmethod
    def get_epex_prices(
        country_code="AT",
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        store_to_file: Path | None = None,
        ) -> pd.Series:
        """Return day-ahead Epex electricity prices in EUR/kWh."""

        if country_code not in EnergyChartsPrice.BIDDING_ZONES:
            raise ValueError(f"Unsupported country_code for Energy-Charts: {country_code}")

        if start_date is None:
            # Take the current time rounded down to the nearest 15 minutes
            start_date = pd.Timestamp.now(tz="Europe/Vienna").floor("15min")
        if end_date is None:
            # Price horizon is max. 1.5 days, so 2 days ensures we get all relevant prices
            end_date = start_date + pd.Timedelta(days=2)

        params = {
            "bzn": EnergyChartsPrice.BIDDING_ZONES[country_code],
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        }
        response = requests.get(EnergyChartsPrice.BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        entries = response.json()["data"]

        index = pd.DatetimeIndex(
            pd.to_datetime([entry["timestamp"] for entry in entries], utc=True)
        ).tz_convert("Europe/Vienna")
        prices = pd.Series(
            [entry["values"]["day_ahead_price"] for entry in entries], index=index
        ).sort_index()
        prices = prices[(prices.index >= start_date) & (prices.index < end_date)]
        # Prices not yet published by the exchange come back as null
        prices = prices.dropna()

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
