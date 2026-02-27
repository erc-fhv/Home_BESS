from pathlib import Path
import json
from entsoe import EntsoePandasClient
import pandas as pd

class DayAheadPrice:
    """Class to read out day-ahead electricity prices from the ENTSO-E Transparency Platform."""

    @staticmethod
    def get_epex_prices(
        country_code="AT",
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        store_to_file: Path | None = None,
        ) -> pd.Series:

        # Set default date range to today if not provided
        if start_date is None:
            start_date = pd.Timestamp.now(tz="Europe/Vienna")
        if end_date is None:
            # Price horizon is max. 1.5 days, so 2 days ensures we get all relevant prices
            end_date = start_date + pd.Timedelta(days=2)

        # Get the API key
        pw_file = Path(__file__).parent.parent.parent.parent / ".json"
        with open(pw_file, encoding="utf-8") as f:
            my_file = json.load(f)
        api_key = my_file["ENTSOE_API_KEY"]

        # Readout day-ahead prices for Austria
        client = EntsoePandasClient(api_key=api_key)
        prices = client.query_day_ahead_prices(
            country_code=country_code,
            start=start_date,
            end=end_date,
        )

        # Check the returned data
        assert isinstance(prices.index, pd.DatetimeIndex), \
            f"Expected DatetimeIndex, got {type(prices.index)}"
        assert prices.index.tz is not None and str(prices.index.tz) == "Europe/Vienna", \
            f"Expected timezone 'Europe/Vienna', got {prices.index.tz}"
        median_freq = prices.index.to_series().diff().median()
        assert median_freq == pd.Timedelta(minutes=15), \
            f"Expected frequency of 15 minutes, got {median_freq}"

        # Convert prices from EUR/MWh to EUR/kWh
        prices = prices / 1000.0

        # Save to CSV if file path is provided
        if store_to_file is not None:
            prices.index.name = "timestamp"
            prices.name = "day_ahead_price_eur_kWh"
            prices.to_csv(store_to_file)

        return prices

    @staticmethod
    def get_prices(
        price_type:str,
        store_to_file: Path | None = None,
        ) -> tuple[pd.Series, pd.Series]:
        """Define sell and buy prices (in EUR/kWh)"""

        price_type = price_type.lower()
        epex_prices = DayAheadPrice.get_epex_prices(store_to_file=store_to_file)

        if price_type == "vkw_dyn":
            price_sell = epex_prices - 0.006  # Subtract 0.6 ct/kWh for selling
            price_buy = epex_prices + 0.0144  # Add 1.44 ct/kWh for buying
        elif price_type == "vkw_fix":
            price_sell = pd.Series(0.09, index=epex_prices.index)
            price_buy  = pd.Series(0.1272, index=epex_prices.index)
        else:
            raise ValueError(f"Unsupported price type: {price_type}")

        return price_sell, price_buy
