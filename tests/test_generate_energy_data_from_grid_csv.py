"""Tests for grid CSV energy data conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.generate_energy_data_from_grid_csv import (
    align_interval_data,
    allocate_self_consumption,
    build_energy_dataframe,
    calculate_self_consumption_kwh,
    calculate_window_days,
    parse_grid_csv,
    parse_timestamp,
)


def write_grid_csv(file_path: Path, rows: list[str]) -> None:
    """Write a grid CSV file with metadata and header."""
    content = [
        "Vertragskonto;123",
        "",
        "Beginn der Messreihe;Ende der Messreihe;Messwert in kWh",
    ]
    content.extend(rows)
    file_path.write_text("\n".join(content), encoding="utf-8")


def test_parse_grid_csv_skips_metadata_and_decimal(tmp_path: Path) -> None:
    """Ensure metadata rows are skipped and decimal commas parsed."""
    csv_path = tmp_path / "import.csv"
    rows = ["01.01.2026 00:00;01.01.2026 00:15;0,130"]
    write_grid_csv(csv_path, rows)

    df = parse_grid_csv(csv_path, "Europe/Vienna")

    assert str(df.index.tz) == "Europe/Vienna"
    assert df.iloc[0]["energy_kwh"] == pytest.approx(0.13)


def test_parse_timestamp_dst_gap() -> None:
    """DST gap should raise a ValueError."""
    tz = pytz.timezone("Europe/Vienna")

    with pytest.raises(ValueError, match="DST"):
        parse_timestamp("28.03.2021 02:30", tz)


def test_parse_timestamp_dst_overlap() -> None:
    """DST overlap should raise a ValueError."""
    tz = pytz.timezone("Europe/Vienna")

    with pytest.raises(ValueError, match="DST"):
        parse_timestamp("31.10.2021 02:30", tz)


def test_calculate_self_consumption_negative() -> None:
    """Negative self-consumption should raise a ValueError."""
    with pytest.raises(ValueError):
        calculate_self_consumption_kwh(365.0, 1.0, 2.0)


def test_allocate_self_consumption_zero_import_interval() -> None:
    """Zero import intervals should receive zero allocation."""
    index = pd.date_range(
        start="2026-01-01 00:00",
        periods=2,
        freq="15min",
        tz="Europe/Vienna",
    )
    import_kwh = pd.Series([0.0, 2.0], index=index)

    allocated = allocate_self_consumption(import_kwh, 3.0)

    assert allocated.iloc[0] == pytest.approx(0.0)
    assert allocated.iloc[1] == pytest.approx(3.0)


def test_build_energy_dataframe_schema() -> None:
    """Output DataFrame should have correct columns and index."""
    index = pd.date_range(
        start="2026-01-01 00:00",
        periods=2,
        freq="15min",
        tz="Europe/Vienna",
    )
    import_kwh = pd.Series([1.0, 1.0], index=index)
    export_kwh = pd.Series([0.5, 0.0], index=index)
    allocated = pd.Series([0.5, 0.5], index=index)

    output_df = build_energy_dataframe(import_kwh, export_kwh, allocated)

    assert list(output_df.columns) == ["Production", "Consumption"]
    assert str(output_df.index.tz) == "Europe/Vienna"
    assert output_df.iloc[0]["Production"] == pytest.approx(1000.0)


def test_align_interval_data_mismatch() -> None:
    """Mismatched intervals should raise a ValueError."""
    index_a = pd.date_range(
        start="2026-01-01 00:00",
        periods=2,
        freq="15min",
        tz="Europe/Vienna",
    )
    index_b = pd.date_range(
        start="2026-01-01 00:15",
        periods=2,
        freq="15min",
        tz="Europe/Vienna",
    )

    import_df = pd.DataFrame(
        {
            "energy_kwh": [0.1, 0.2],
            "end_time": index_a + pd.Timedelta(minutes=15),
        },
        index=index_a,
    )
    export_df = pd.DataFrame(
        {
            "energy_kwh": [0.1, 0.2],
            "end_time": index_b + pd.Timedelta(minutes=15),
        },
        index=index_b,
    )

    with pytest.raises(ValueError, match="align"):
        align_interval_data(import_df, export_df)


def test_calculate_window_days() -> None:
    """Window duration should match interval range."""
    index = pd.date_range(
        start="2026-01-01 00:00",
        periods=2,
        freq="15min",
        tz="Europe/Vienna",
    )
    end_times = pd.Series(index + pd.Timedelta(minutes=15), index=index)

    window_days = calculate_window_days(index, end_times)

    assert window_days == pytest.approx(0.0208333, rel=1e-3)
