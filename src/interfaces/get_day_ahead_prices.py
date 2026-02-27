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
            start_date = pd.Timestamp.now(tz="Europe/Vienna").normalize()
        if end_date is None:
            # Price horizon is max. 1.5 days, so 2 days ensures we get all relevant prices
            end_date = start_date + pd.Timedelta(days=2)

        # Get the API key
        pw_file = Path.cwd().parent.parent / ".json"
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
        assert prices.index.tz is not None and prices.index.tz.zone == "Europe/Vienna", \
            f"Expected timezone 'Europe/Vienna', got {prices.index.tz}"
        assert prices.index.freq == pd.Timedelta(minutes=15), \
            f"Expected frequency of 15 minutes, got {prices.index.freq}"

        # Convert prices from EUR/MWh to EUR/kWh
        prices = prices["day_ahead_price_EUR_MWh"] / 1000

        # Save to CSV if file path is provided
        if store_to_file is not None:
            prices.index.name = "timestamp"
            prices.name = "day_ahead_price_EUR_kWh"
            prices.to_csv(store_to_file)

        return prices

    @staticmethod
    def get_prices(price_type:str) -> tuple[pd.Series, pd.Series]:
        """Define sell and buy prices (in EUR/kWh)"""

        price_type = price_type.lower()
        epex_prices = DayAheadPrice.get_epex_prices()

        if price_type == "vkw_dyn":
            price_sell = epex_prices - 0.006  # Subtract 6 ct/kWh for selling
            price_buy = epex_prices + 0.0144  # Add 14.4 ct/kWh for buying
        if price_type == "vkw_fix":
            price_sell = pd.Series(0.09, index=epex_prices.index)
            price_buy  = pd.Series(0.1272, index=epex_prices.index)
        else:
            raise ValueError(f"Unsupported price type: {price_type}")

        return price_sell, price_buy
