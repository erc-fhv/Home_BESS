"""Generate energy data from grid import/export CSVs.

This script converts utility interval CSVs for grid import and export into
`energy_data_01.csv` with Production and Consumption in Wh at 15-minute
resolution.

Examples
--------
Edit the configuration in the `__main__` section to point at your
import/export CSVs and set the annual PV assumption.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

DEFAULT_ANNUAL_PV_KWH = 8000.0
TZ_NAME = "Europe/Vienna"
DEFAULT_OUTPUT = Path("data/energy_data_01.csv")


def read_csv_text(file_path: Path) -> str:
    """Read CSV contents with a fallback for common encodings.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    str
        Decoded CSV contents as text.

    Raises
    ------
    ValueError
        If the file cannot be decoded with supported encodings.
    """
    encodings = (
        "utf-8-sig",
        "utf-16-le",
        "utf-16-be",
        "utf-16",
        "latin-1",
    )

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(
        "Unable to decode CSV. Supported encodings are utf-8-sig, utf-16, utf-16-le, utf-16-be, and latin-1."
    )


def clean_header_cell(value: str) -> str:
    """Normalize header cell values for comparison.

    Parameters
    ----------
    value : str
        Raw cell value.

    Returns
    -------
    str
        Normalized value with BOM artifacts removed.
    """
    stripped = value.strip()
    return stripped.replace("\ufeff", "").replace("ï»¿", "")


def parse_decimal_comma(value: str) -> float:
    """Parse a decimal comma number into a float.

    Parameters
    ----------
    value : str
        Value using a decimal comma separator.

    Returns
    -------
    float
        Parsed floating-point number.
    """
    return float(value.replace(",", "."))


def parse_timestamp(timestamp_str: str, tz: pytz.BaseTzInfo) -> datetime:
    """Parse a local timestamp and localize it to Europe/Vienna.

    Parameters
    ----------
    timestamp_str : str
        Timestamp in DD.MM.YYYY HH:MM format.
    tz : pytz.BaseTzInfo
        Timezone for localization.

    Returns
    -------
    datetime
        Timezone-aware timestamp.

    Raises
    ------
    ValueError
        If the timestamp is ambiguous or nonexistent due to DST.
    """
    naive_time = datetime.strptime(timestamp_str, "%d.%m.%Y %H:%M")  # noqa: DTZ007

    try:
        return tz.localize(naive_time, is_dst=None)
    except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError) as exc:
        message = f"Timestamp is invalid due to DST ambiguity or gap: {timestamp_str}"
        raise ValueError(message) from exc


def parse_grid_csv(file_path: Path, tz_name: str) -> pd.DataFrame:
    """Parse a grid import/export CSV file into a structured DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.
    tz_name : str
        Timezone name for timestamp localization.

    Returns
    -------
    pd.DataFrame
        DataFrame with index as interval start and columns:
        - end_time: timezone-aware interval end timestamps
        - energy_kwh: energy values in kWh
    """
    tz = pytz.timezone(tz_name)
    records: list[dict[str, object]] = []
    header_found = False

    content = read_csv_text(file_path)
    reader = csv.reader(io.StringIO(content), delimiter=";")

    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        if clean_header_cell(row[0]) == "Beginn der Messreihe":
            header_found = True
            break

    if not header_found:
        raise ValueError(f"CSV header 'Beginn der Messreihe' not found in {file_path}")

    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < 3:
            raise ValueError(f"CSV row must have start, end, and energy values in {file_path}: {row}")

        start_str = row[0].strip()
        end_str = row[1].strip()
        energy_str = row[2].strip()
        start_time = parse_timestamp(start_str, tz)
        end_time = parse_timestamp(end_str, tz)
        energy_kwh = parse_decimal_comma(energy_str)

        records.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "energy_kwh": energy_kwh,
            }
        )

    if not records:
        raise ValueError(f"No interval rows found in {file_path}.")

    df = pd.DataFrame(records).set_index("start_time")
    df.index = pd.DatetimeIndex(df.index)
    return df


def align_interval_data(import_df: pd.DataFrame, export_df: pd.DataFrame) -> pd.DataFrame:
    """Align import and export data on interval start timestamps.

    Parameters
    ----------
    import_df : pd.DataFrame
        Import data with interval start index and energy_kwh column.
    export_df : pd.DataFrame
        Export data with interval start index and energy_kwh column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns import_kwh, export_kwh, end_time.

    Raises
    ------
    ValueError
        If intervals are not aligned between files.
    """
    if not import_df.index.equals(export_df.index):
        raise ValueError("Import and export intervals do not align.")

    if not import_df["end_time"].equals(export_df["end_time"]):
        raise ValueError("Import and export interval end times do not match.")

    return pd.DataFrame(
        {
            "import_kwh": import_df["energy_kwh"],
            "export_kwh": export_df["energy_kwh"],
            "end_time": import_df["end_time"],
        },
        index=import_df.index,
    )


def calculate_window_days(start_times: pd.DatetimeIndex, end_times: pd.Series) -> float:
    """Calculate window duration in days from interval timestamps.

    Parameters
    ----------
    start_times : pd.DatetimeIndex
        Interval start timestamps.
    end_times : pd.Series
        Interval end timestamps.

    Returns
    -------
    float
        Duration in days.
    """
    window_start = start_times.min()
    window_end = end_times.max()
    duration = window_end - window_start
    return duration.total_seconds() / 86400.0


def calculate_self_consumption_kwh(annual_pv_kwh: float, window_days: float, export_kwh: float) -> tuple[float, float]:
    """Compute window PV and self-consumed PV in kWh.

    Parameters
    ----------
    annual_pv_kwh : float
        Annual PV production assumption in kWh.
    window_days : float
        Duration of the data window in days.
    export_kwh : float
        Total grid export in kWh.

    Returns
    -------
    tuple[float, float]
        Window PV total (kWh) and self-consumed PV (kWh).

    Raises
    ------
    ValueError
        If export exceeds the inferred window PV total.
    """
    window_pv_kwh = annual_pv_kwh * (window_days / 365.0)
    self_consumed_kwh = window_pv_kwh - export_kwh

    if self_consumed_kwh < 0:
        raise ValueError("Export exceeds window PV total. Adjust annual PV assumption.")

    return window_pv_kwh, self_consumed_kwh


def allocate_self_consumption(import_kwh: pd.Series, self_consumed_kwh: float) -> pd.Series:
    """Allocate self-consumed PV across intervals proportional to import.

    Parameters
    ----------
    import_kwh : pd.Series
        Interval grid import in kWh.
    self_consumed_kwh : float
        Total self-consumed PV in kWh.

    Returns
    -------
    pd.Series
        Allocated self-consumption per interval in kWh.
    """
    total_import_kwh = import_kwh.sum()

    if total_import_kwh > 0:
        weights = import_kwh / total_import_kwh
    else:
        weights = import_kwh * 0.0

    allocated_kwh = self_consumed_kwh * weights
    allocated_kwh[import_kwh <= 0] = 0.0
    return allocated_kwh


def build_energy_dataframe(
    import_kwh: pd.Series,
    export_kwh: pd.Series,
    allocated_self_kwh: pd.Series,
) -> pd.DataFrame:
    """Build the output energy DataFrame in Wh.

    Parameters
    ----------
    import_kwh : pd.Series
        Interval grid import in kWh.
    export_kwh : pd.Series
        Interval grid export in kWh.
    allocated_self_kwh : pd.Series
        Allocated self-consumption per interval in kWh.

    Returns
    -------
    pd.DataFrame
        DataFrame with Production and Consumption in Wh.
    """
    production_wh = (export_kwh + allocated_self_kwh) * 1000.0
    consumption_wh = (import_kwh + allocated_self_kwh) * 1000.0

    return pd.DataFrame(
        {
            "Production": production_wh,
            "Consumption": consumption_wh,
        },
        index=import_kwh.index,
    )


def validate_energy_totals(
    import_kwh: pd.Series,
    export_kwh: pd.Series,
    allocated_self_kwh: pd.Series,
    self_consumed_kwh: float,
) -> dict[str, float]:
    """Validate energy totals and return summary metrics.

    Parameters
    ----------
    import_kwh : pd.Series
        Interval grid import in kWh.
    export_kwh : pd.Series
        Interval grid export in kWh.
    allocated_self_kwh : pd.Series
        Allocated self-consumption per interval in kWh.
    self_consumed_kwh : float
        Total self-consumed PV in kWh.

    Returns
    -------
    dict[str, float]
        Summary totals for import, export, production, and consumption.

    Raises
    ------
    ValueError
        If energy balance checks fail.
    """
    total_import_kwh = import_kwh.sum()
    total_export_kwh = export_kwh.sum()
    total_production_kwh = (export_kwh + allocated_self_kwh).sum()
    total_consumption_kwh = (import_kwh + allocated_self_kwh).sum()

    tolerance_kwh = 1e-6
    production_expected = total_export_kwh + self_consumed_kwh
    consumption_expected = total_import_kwh + self_consumed_kwh

    if abs(total_production_kwh - production_expected) > tolerance_kwh:
        raise ValueError("Production total does not balance.")

    if abs(total_consumption_kwh - consumption_expected) > tolerance_kwh:
        raise ValueError("Consumption total does not balance.")

    return {
        "import_kwh": total_import_kwh,
        "export_kwh": total_export_kwh,
        "production_kwh": total_production_kwh,
        "consumption_kwh": total_consumption_kwh,
    }


def run_conversion(
    energy_from_grid: Path,
    energy_to_grid: Path,
    output_path: Path,
    annual_pv_kwh: float,
) -> None:
    """Run the CSV conversion and write energy data output.

    Parameters
    ----------
    energy_from_grid : Path
        Path to grid import CSV file.
    energy_to_grid : Path
        Path to grid export CSV file.
    output_path : Path
        Output CSV path.
    annual_pv_kwh : float
        Annual PV production in kWh.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import_df = parse_grid_csv(energy_from_grid, TZ_NAME)
    export_df = parse_grid_csv(energy_to_grid, TZ_NAME)
    intervals = align_interval_data(import_df, export_df)

    window_days = calculate_window_days(intervals.index, intervals["end_time"])
    export_total_kwh = intervals["export_kwh"].sum()

    window_pv_kwh, self_consumed_kwh = calculate_self_consumption_kwh(annual_pv_kwh, window_days, export_total_kwh)

    allocated_self_kwh = allocate_self_consumption(intervals["import_kwh"], self_consumed_kwh)

    output_df = build_energy_dataframe(
        intervals["import_kwh"],
        intervals["export_kwh"],
        allocated_self_kwh,
    )

    totals = validate_energy_totals(
        intervals["import_kwh"],
        intervals["export_kwh"],
        allocated_self_kwh,
        self_consumed_kwh,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path)

    logger.info("Conversion complete.")
    logger.info("Output: %s", output_path)
    logger.info(
        "Date range: %s to %s",
        intervals.index.min(),
        intervals.index.max(),
    )
    logger.info("Annual PV parameter: %.1f kWh", annual_pv_kwh)
    logger.info("Window PV total: %.3f kWh", window_pv_kwh)
    logger.info("Self-consumed PV: %.3f kWh", self_consumed_kwh)
    logger.info("Total import: %.3f kWh", totals["import_kwh"])
    logger.info("Total export: %.3f kWh", totals["export_kwh"])
    logger.info("Total production: %.3f kWh", totals["production_kwh"])
    logger.info("Total consumption: %.3f kWh", totals["consumption_kwh"])


if __name__ == "__main__":
    ENERGY_FROM_GRID = Path("data/energy_from_grid_01.csv")
    ENERGY_TO_GRID = Path("data/energy_to_grid_01.csv")
    OUTPUT_PATH = DEFAULT_OUTPUT
    ANNUAL_PV_KWH = DEFAULT_ANNUAL_PV_KWH

    run_conversion(
        energy_from_grid=ENERGY_FROM_GRID,
        energy_to_grid=ENERGY_TO_GRID,
        output_path=OUTPUT_PATH,
        annual_pv_kwh=ANNUAL_PV_KWH,
    )
