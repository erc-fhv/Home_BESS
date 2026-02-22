from pathlib import Path
import json
from entsoe import EntsoePandasClient
import pandas as pd

class DayAheadPrice:
    """Class to read out day-ahead electricity prices from the ENTSO-E Transparency Platform."""

    @staticmethod
    def get_prices(
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

        # Save to CSV if file path is provided
        if store_to_file is not None:
            prices.index.name = "timestamp"
            prices.name = "day_ahead_price_EUR_MWh"
            prices.to_csv(store_to_file)

        return prices
