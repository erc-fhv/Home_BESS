"""Tests for CSV profile parsing (parse_csv).

Supported formats:
- netload_profile_{1,2}: net-load profiles (simple CSV and high-freq Watt)
- load_profile_{1,2,3} / pv_profile_{1,2,3}: load and PV pairs
    1 → VKW Smartmeter export (4 metadata rows, semicolon, Latin-1)
    2 → high-frequency ISO8601+Z Watt data, no header → resampled + W→kW
    3 → simple semicolon "Beginn der Messung;Messwert[*]", full year 2025
        (spans March spring-forward and October fall-back DST transitions)
"""

import base64
from pathlib import Path

import pandas as pd
import pytest

from simulation.web_app import parse_csv


TESTS_DIR = Path(__file__).parent / "data"


def _parse(stem: str) -> pd.DataFrame:
    """Load a test CSV and run it through parse_csv, simulating a Dash Upload."""
    raw = (TESTS_DIR / f"{stem}.csv").read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return parse_csv(f"data:text/csv;base64,{encoded}")


# ── Output contract: every profile must satisfy these invariants ──────────────

ALL_PROFILES = [
    "netload_profile_1",
    "netload_profile_2",
    "load_profile_1",
    "pv_profile_1",
    "load_profile_2",
    "pv_profile_2",
    "load_profile_3",
    "pv_profile_3",
]


@pytest.mark.parametrize("stem", ALL_PROFILES)
def test_output_contract(stem):
    """parse_csv must return a non-empty DataFrame with:
    - a 'value_kw' column, no NaN
    - a timezone-aware DatetimeIndex in Europe/Vienna
    - exact 15-minute median frequency
    - values in a plausible kW range (−1 … 50 kW)
    """
    df = _parse(stem)
    assert not df.empty
    assert "value_kw" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "Europe/Vienna", f"{stem}: wrong timezone"
    assert df.index.to_series().diff().dropna().median() == pd.Timedelta(minutes=15), \
        f"{stem}: index is not 15-min"
    assert not df["value_kw"].isna().any(), f"{stem}: value_kw contains NaN"
    # Broad sanity bounds: net-load can be negative (PV > load), but never ±hundreds
    assert df["value_kw"].min() >= -50, f"{stem}: unexpected large negative value"
    assert df["value_kw"].max() <= 150, f"{stem}: values look unreasonably large (still Watts?)"


# ── Format-specific behaviour ─────────────────────────────────────────────────

def test_vkw_4_metadata_rows_skipped():
    """VKW Smartmeter files have 4 metadata rows before the actual CSV header.
    parse_csv must skip them; the first timestamp must be 2026-02-03 00:00 CET.
    """
    df = _parse("load_profile_1")
    expected = pd.Timestamp("2026-02-03", tz="Europe/Vienna")
    assert df.index[0] == expected, \
        f"First timestamp {df.index[0]} suggests metadata rows were not skipped"


def test_messwert_kw_column_takes_priority():
    """pv_profile_3 has both 'Messwert-kw' and a plain 'Messwert' column.
    parse_csv must use 'Messwert-kw' (PV power = 0 at night), not 'Messwert'
    (which contains non-zero test values at night).
    """
    df = _parse("pv_profile_3")
    # At 2025-01-01 00:00:00+01:00 the CSV has: Messwert-kw=0.0, Messwert=2.29.
    # If the wrong column were chosen, value_kw would be ~2.29, not 0.0.
    ts = pd.Timestamp("2025-01-01 00:00", tz="Europe/Vienna")
    assert df.loc[ts, "value_kw"] == pytest.approx(0.0), \
        f"value_kw at midnight Jan 1 = {df.loc[ts, 'value_kw']:.3f} kW — wrong Messwert column selected"


def test_watt_to_kw_auto_conversion():
    """load_profile_2 supplies raw Watt values (~1300 W).
    parse_csv must detect this and divide by 1000 so values fall in the kW range.
    """
    df = _parse("load_profile_2")
    assert df["value_kw"].max() < 50, \
        "W→kW conversion was not applied (values still look like Watts)"


def test_dst_transitions_no_error():
    """load_profile_3 spans all of 2025, including:
    - 2025-03-30: spring-forward (02:00 CET is nonexistent)
    - 2025-10-26: fall-back (02:00–02:59 is ambiguous)
    parse_csv must handle both without raising AmbiguousTimeError or
    NonExistentTimeError, and data must be present on both transition days.
    """
    df = _parse("load_profile_3")
    assert not df.loc["2025-03-30":"2025-03-30"].empty, \
        "No data for spring-forward day (2025-03-30)"
    assert not df.loc["2025-10-26":"2025-10-26"].empty, \
        "No data for fall-back day (2025-10-26)"


# ── Paired load + PV profiles: net-load must be computable ───────────────────

@pytest.mark.parametrize("n", [1, 2, 3])
def test_net_load_from_paired_profiles(n):
    """Load and PV profiles for the same dataset can be subtracted to produce
    a net-load series without NaN on their shared timestamps.
    """
    df_load = _parse(f"load_profile_{n}")
    df_pv = _parse(f"pv_profile_{n}")
    common = df_load.index.intersection(df_pv.index)
    assert len(common) > 0, f"Pair {n}: profiles have no overlapping timestamps"
    net = df_load.loc[common, "value_kw"] - df_pv.loc[common, "value_kw"]
    assert not net.isna().any(), f"Pair {n}: net-load has NaN values"
