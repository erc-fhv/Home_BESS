"""Generate realistic synthetic energy data for 2025 BESS simulation.

This script generates synthetic PV production and household consumption data
for Austria (Vorarlberg region) with realistic seasonal and daily variations
at 15-minute resolution for the entire year 2025.

Output: data/energy_data.csv with columns [Production, Consumption] in Wh.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

LOCATION_LAT = 47.25
LOCATION_LON = 9.75
LOCATION_ALT = 405.0
LOCATION_NAME = "Vorarlberg, Austria"

PV_CAPACITY_KW = 10.0
CONSUMPTION_DAILY_KWH = 25.0

YEAR = 2025
TZ_NAME = "Europe/Vienna"
OUTPUT_FILE = Path("data/energy_data.csv")


def generate_pv_production(
    timestamps_utc: pd.DatetimeIndex,
    pv_capacity_kw: float,
) -> pd.Series:
    """Generate realistic PV production data using solar position.

    Parameters
    ----------
    timestamps_utc : pd.DatetimeIndex
        Timestamps in UTC timezone.
    pv_capacity_kw : float
        Installed PV capacity in kW.

    Returns
    -------
    pd.Series
        PV production in Wh with corresponding timestamps.
    """
    location = pvlib.location.Location(
        latitude=LOCATION_LAT,
        longitude=LOCATION_LON,
        tz=TZ_NAME,
        altitude=LOCATION_ALT,
        name=LOCATION_NAME,
    )

    clearsky = location.get_clearsky(timestamps_utc, model="ineichen")

    np.random.seed(42)
    cloud_variation = np.random.uniform(0.7, 1.0, len(timestamps_utc))
    ghi_wm2 = clearsky["ghi"].values * cloud_variation
    ghi_wm2 = np.maximum(ghi_wm2, 0)

    dc_power_kw = (pv_capacity_kw / 1000.0) * ghi_wm2 * 0.85
    pv_production_kwh = dc_power_kw * (15 / 60)
    pv_production_wh = pv_production_kwh * 1000

    return pd.Series(pv_production_wh, index=timestamps_utc, name="Production")


def generate_consumption(
    timestamps_vienna: pd.DatetimeIndex,
    daily_consumption_kwh: float,
) -> pd.Series:
    """Generate realistic household consumption with seasonal and daily variation.

    Parameters
    ----------
    timestamps_vienna : pd.DatetimeIndex
        Timestamps in Europe/Vienna timezone.
    daily_consumption_kwh : float
        Target average daily consumption in kWh.

    Returns
    -------
    pd.Series
        Consumption in Wh with corresponding timestamps.
    """
    hour_of_day = timestamps_vienna.hour
    minute_of_day = timestamps_vienna.minute
    day_of_year = timestamps_vienna.dayofyear

    morning_peak = 3.5 * np.exp(-((hour_of_day + minute_of_day / 60 - 7.5) ** 2) / 2)
    evening_peak = 3.0 * np.exp(-((hour_of_day + minute_of_day / 60 - 19) ** 2) / 3)
    night_base = 0.3

    daily_profile = night_base + np.maximum(morning_peak, 0) + np.maximum(evening_peak, 0)

    winter_heating_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 1) / 365)
    seasonal_factor = 0.7 + 0.3 * winter_heating_factor

    np.random.seed(43)
    noise = np.random.normal(1.0, 0.08, len(timestamps_vienna))

    avg_intervals_per_day = 96
    target_kw_per_interval = daily_consumption_kwh / (avg_intervals_per_day * 15 / 60)

    consumption_kw = target_kw_per_interval * daily_profile * seasonal_factor * noise
    consumption_kw = np.maximum(consumption_kw, 0)

    consumption_kwh = consumption_kw * (15 / 60)
    consumption_wh = consumption_kwh * 1000

    return pd.Series(consumption_wh, index=timestamps_vienna, name="Consumption")


def main() -> None:
    """Generate and save synthetic energy data for 2025."""
    print(f"Generating synthetic energy data for {YEAR}...")

    timestamps_vienna = pd.date_range(
        start=f"{YEAR}-01-01",
        end=f"{YEAR}-12-31 23:45",
        freq="15min",
        tz=TZ_NAME,
    )

    timestamps_utc = timestamps_vienna.tz_convert("UTC")

    print(f"  - Computing PV production ({len(timestamps_vienna)} intervals)...")
    pv_production = generate_pv_production(timestamps_utc, PV_CAPACITY_KW)

    print("  - Computing household consumption...")
    consumption = generate_consumption(timestamps_vienna, CONSUMPTION_DAILY_KWH)

    df_energy = pd.DataFrame(
        {
            "Production": pv_production.values,
            "Consumption": consumption.values,
        },
        index=timestamps_vienna,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_energy.to_csv(OUTPUT_FILE)

    daily_pv_kwh = df_energy["Production"].resample("D").sum() / 1000
    daily_consumption_kwh = df_energy["Consumption"].resample("D").sum() / 1000

    print("\nData generation complete!")
    print(f"  - Output: {OUTPUT_FILE}")
    print(f"  - Records: {len(df_energy)}")
    print(f"  - Date range: {timestamps_vienna[0]} to {timestamps_vienna[-1]}")
    print("\nAnnual summary:")
    print(f"  - PV production: {daily_pv_kwh.sum():.1f} kWh/year")
    print(f"  - Consumption: {daily_consumption_kwh.sum():.1f} kWh/year")
    print(f"  - Daily avg PV: {daily_pv_kwh.mean():.1f} kWh")
    print(f"  - Daily avg consumption: {daily_consumption_kwh.mean():.1f} kWh")
    print(f"  - Min daily PV: {daily_pv_kwh.min():.1f} kWh (winter)")
    print(f"  - Max daily PV: {daily_pv_kwh.max():.1f} kWh (summer)")


if __name__ == "__main__":
    main()
