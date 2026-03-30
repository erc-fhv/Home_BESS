"""Tests for CSV profile parsing (parse_csv) and simulation compatibility.

Covers:
- netload_profile_1.csv  (simple CSV with header, kW values, Vienna TZ)
- netload_profile_2.csv  (high-frequency format, no header, ISO8601+Z, Watt)
- load_profile_1.csv + pv_profile_1.csv  (VKW Smartmeter format, semicolons, Latin-1)
"""

import base64
from pathlib import Path

import pandas as pd
import pytest

from simulation.web_app import parse_csv


TESTS_DIR = Path(__file__).parent


def _to_upload_contents(file_path: Path) -> str:
    """Simulate a Dash Upload component's `contents` string (data URI)."""
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:text/csv;base64,{encoded}"


# ── netload_profile_1.csv (simple CSV) ───────────────────────────────────────

class TestNetloadProfile1:
    @pytest.fixture()
    def df(self):
        contents = _to_upload_contents(TESTS_DIR / "netload_profile_1.csv")
        return parse_csv(contents)

    def test_not_empty(self, df):
        assert not df.empty

    def test_has_value_kw_column(self, df):
        assert "value_kw" in df.columns

    def test_index_is_datetime(self, df):
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_tz_is_vienna(self, df):
        assert df.index.tz is not None
        assert str(df.index.tz) == "Europe/Vienna"

    def test_15min_frequency(self, df):
        freq = df.index.to_series().diff().dropna().median()
        assert freq == pd.Timedelta(minutes=15)

    def test_values_plausible(self, df):
        assert df["value_kw"].min() >= -50, "Unexpectedly large negative value"
        assert df["value_kw"].max() <= 50, "Unexpectedly large positive value"


# ── netload_profile_2.csv (high-frequency / Watt / no header) ────────────────

class TestNetloadProfile2:
    @pytest.fixture()
    def df(self):
        contents = _to_upload_contents(TESTS_DIR / "netload_profile_2.csv")
        return parse_csv(contents)

    def test_not_empty(self, df):
        assert not df.empty

    def test_has_value_kw_column(self, df):
        assert "value_kw" in df.columns

    def test_index_is_datetime(self, df):
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_tz_is_vienna(self, df):
        assert df.index.tz is not None
        assert str(df.index.tz) == "Europe/Vienna"

    def test_15min_frequency(self, df):
        freq = df.index.to_series().diff().dropna().median()
        assert freq == pd.Timedelta(minutes=15)

    def test_values_in_kw(self, df):
        # Data has ~15s intervals resampled to 15min, values ~1-3 kW
        assert df["value_kw"].max() < 50, "Values seem to still be in Watts"

    def test_values_plausible(self, df):
        assert df["value_kw"].min() >= -50
        assert df["value_kw"].max() <= 50


# ── load_profile_1.csv + pv_profile_1.csv (VKW Smartmeter) ──────────────────

class TestVKWProfiles:
    @pytest.fixture()
    def df_load(self):
        contents = _to_upload_contents(TESTS_DIR / "load_profile_1.csv")
        return parse_csv(contents)

    @pytest.fixture()
    def df_pv(self):
        contents = _to_upload_contents(TESTS_DIR / "pv_profile_1.csv")
        return parse_csv(contents)

    def test_load_not_empty(self, df_load):
        assert not df_load.empty

    def test_pv_not_empty(self, df_pv):
        assert not df_pv.empty

    def test_load_has_value_kw(self, df_load):
        assert "value_kw" in df_load.columns

    def test_pv_has_value_kw(self, df_pv):
        assert "value_kw" in df_pv.columns

    def test_load_tz_vienna(self, df_load):
        assert df_load.index.tz is not None
        assert str(df_load.index.tz) == "Europe/Vienna"

    def test_pv_tz_vienna(self, df_pv):
        assert df_pv.index.tz is not None
        assert str(df_pv.index.tz) == "Europe/Vienna"

    def test_load_15min_frequency(self, df_load):
        freq = df_load.index.to_series().diff().dropna().median()
        assert freq == pd.Timedelta(minutes=15)

    def test_pv_15min_frequency(self, df_pv):
        freq = df_pv.index.to_series().diff().dropna().median()
        assert freq == pd.Timedelta(minutes=15)

    def test_indexes_match(self, df_load, df_pv):
        assert df_load.index.equals(df_pv.index), \
            "Load and PV profiles must cover the same time range"

    def test_net_load_computable(self, df_load, df_pv):
        net = df_load["value_kw"] - df_pv["value_kw"]
        assert not net.isna().any(), "Net load has NaNs after subtraction"

    def test_load_values_plausible(self, df_load):
        assert df_load["value_kw"].min() >= -1, "Load should not be strongly negative"
        assert df_load["value_kw"].max() <= 50

    def test_pv_values_plausible(self, df_pv):
        assert df_pv["value_kw"].min() >= -1
        assert df_pv["value_kw"].max() <= 50
