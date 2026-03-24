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
        """Return day-ahead Epex electricity prices in EUR/kWh."""

        if start_date is None:
            # Take the current time rounded down to the nearest 15 minutes
            start_date = pd.Timestamp.now(tz="Europe/Vienna").floor("15min")
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

    @staticmethod
    def get_prices(
        price_type:str,
        store_to_file: Path | None = None,
        start_date: pd.Timestamp | None = None,
        epex_offset_buy: float = 0.0144,
        epex_offset_sell: float = 0.006,
        grid_fee: float = 0.06,
        vat: float = 0.20,
        fix_price_buy: float = 0.1272,
        fix_price_sell: float = 0.09,
        ) -> tuple[pd.Series, pd.Series]:
        """Define sell and buy prices (in EUR/kWh)"""

        price_type = price_type.lower()
        epex_prices = DayAheadPrice.get_epex_prices(store_to_file=store_to_file,
            start_date=start_date)

        if price_type == "vkw_dyn":
            # VKW dynamische Preise in EUR/kWh
            price_sell = epex_prices - epex_offset_sell
            price_buy  = (epex_prices + epex_offset_buy + grid_fee) * (1 + vat)
        elif price_type == "vkw_fix":
            price_sell = pd.Series(fix_price_sell, index=epex_prices.index)
            price_buy  = pd.Series((fix_price_buy + grid_fee) * (1 + vat),
                index=epex_prices.index)

        else:
            raise ValueError(f"Unsupported price type: {price_type}")

        return price_sell, price_buy
