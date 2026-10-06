from pathlib import Path
import json
from entsoe import EntsoePandasClient
import pandas as pd

from interfaces.get_day_ahead_prices_awattar import AwattarPrice
from interfaces.get_day_ahead_prices_energycharts import EnergyChartsPrice

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
    def get_epex_prices_with_fallback(
        epex_sources: list[str],
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        store_to_file: Path | None = None,
        ) -> pd.Series:
        """Return day-ahead Epex prices in EUR/kWh from the first source in epex_sources
        that works. A RuntimeError is raised only if all sources fail."""

        source_functions = {
            "entsoe": DayAheadPrice.get_epex_prices,
            "awattar": AwattarPrice.get_epex_prices,
            "energycharts": EnergyChartsPrice.get_epex_prices,
        }
        errors = []
        for epex_source in epex_sources:
            epex_source = epex_source.lower()
            if epex_source not in source_functions:
                raise ValueError(f"Unsupported epex_source: {epex_source}")
            try:
                prices = source_functions[epex_source](
                    start_date=start_date, end_date=end_date, store_to_file=store_to_file)
                if prices.empty:
                    raise ValueError("No prices returned.")
                return prices
            except Exception as e:
                print(f"Price source '{epex_source}' failed: {e}")
                errors.append(f"{epex_source}: {type(e).__name__}: {e}")

        raise RuntimeError("All price sources failed: " + "; ".join(errors))

    @staticmethod
    def get_prices(
        price_type:str,
        store_to_file: Path | None = None,
        start_date: pd.Timestamp | None = None,
        epex_sources: list[str] = ("energycharts", "entsoe"),
        epex_offset_buy: float = 0.0144,
        epex_offset_sell: float = 0.006,
        grid_fee: float = 0.06,
        vat: float = 0.20,
        fix_price_buy: float = 0.1272,
        fix_price_sell: float = 0.09,
        ) -> tuple[pd.Series, pd.Series]:
        """Define sell and buy prices (in EUR/kWh)"""

        price_type = price_type.lower()
        epex_prices = DayAheadPrice.get_epex_prices_with_fallback(
            epex_sources, start_date=start_date, store_to_file=store_to_file)

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


class CachedDayAheadPrice:
    """Cache of sell and buy prices that is refreshed at startup and from 12:00 on
    until the next day's prices are available. Afterwards the cached prices are used
    until 12:00 of the next day."""

    def __init__(self, min_horizon: pd.Timedelta = pd.Timedelta(hours=6)) -> None:
        self.min_horizon = min_horizon
        self.price_sell = pd.Series(dtype=float)
        self.price_buy = pd.Series(dtype=float)
        self.fetch_error = None

    def get_prices(
        self,
        current_time: pd.Timestamp,
        epex_sources: list[str],
        price_type: str = "vkw_dyn",
        ) -> tuple[pd.Series, pd.Series]:
        """Return sell and buy prices (in EUR/kWh) starting at the current 15-min interval.

        A RuntimeError is raised only if the remaining price horizon is shorter than
        min_horizon."""

        if self.price_sell.empty \
                or self.price_sell.index[-1] < self.required_price_end(current_time):
            try:
                price_sell, price_buy = DayAheadPrice.get_prices(
                    price_type, start_date=current_time.floor("15min"),
                    epex_sources=epex_sources)
                if self.price_sell.empty or price_sell.index[-1] >= self.price_sell.index[-1]:
                    self.price_sell, self.price_buy = price_sell, price_buy
                self.fetch_error = None
            except Exception as e:
                print(f"Fetching new prices failed, using cached prices. Error: {e}")
                self.fetch_error = e

        if self.price_sell.empty:
            raise RuntimeError("No price data available.") from self.fetch_error
        if self.price_sell.index[-1] - current_time < self.min_horizon:
            raise RuntimeError("Price horizon is shorter than " + \
                f"{self.min_horizon / pd.Timedelta(hours=1):g} hours and " + \
                "no new price data available.") from self.fetch_error

        price_sell = self.price_sell[current_time.floor("15min"):]
        price_buy = self.price_buy[current_time.floor("15min"):]

        assert isinstance(price_sell.index, pd.DatetimeIndex)
        assert current_time - pd.Timedelta(minutes=15) < price_sell.index[0] <= current_time, \
            f"Act time: {current_time}, price start: {price_sell.index[0]}"

        return price_sell, price_buy

    @staticmethod
    def required_price_end(current_time: pd.Timestamp) -> pd.Timestamp:
        """Return the timestamp up to which prices should be available: the end of the
        current day before 12:00 and the end of the next day from 12:00 on."""

        day = current_time.normalize()
        if current_time.hour >= 12:
            day = day + pd.DateOffset(days=1)
        # Last price interval starts at 23:00 (hourly) or 23:45 (15-min resolution)
        return day.replace(hour=23)
