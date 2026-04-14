import base64
import collections
import copy
from io import StringIO
import threading
import uuid

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback_context, State, no_update
from dash.exceptions import PreventUpdate
from flask_socketio import SocketIO, join_room

from simulation.bess_simulation import Bess

# Max possible sessions. Needed for server-side caching, in order to
# have higher performance for slow devices.
_MAX_SESSIONS = 100


class _BoundedCache(collections.OrderedDict):
    """OrderedDict mit LRU-Eviction: älteste Einträge werden entfernt wenn maxsize überschritten."""

    def __init__(self, maxsize: int = _MAX_SESSIONS):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)

    def get_or_create(self, key, factory=dict):
        """Wie setdefault, aber bewegt den Key ans Ende (= zuletzt benutzt)."""
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = factory()
        self[key] = val
        return val

# ── Styling-Konstanten ───────────────────────────────────────────────────────
COLOR = {
    "header":     "#1e293b",
    "accent":     "#3b82f6",
    "card":       "#ffffff",
    "bg":         "#f1f5f9",
    "border":     "#e2e8f0",
    "text":       "#334155",
    "text_light": "#64748b",
}

CARD = {
    "backgroundColor": COLOR["card"],
    "borderRadius": "12px",
    "padding": "18px 22px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
    "border": f"1px solid {COLOR['border']}",
    "flex": "1",
    "minWidth": "200px",
}

CARD_TITLE = {
    "fontWeight": "700",
    "fontSize": "11px",
    "color": COLOR["text_light"],
    "textTransform": "uppercase",
    "letterSpacing": "1px",
    "marginBottom": "10px",
}

BTN = {
    "padding": "8px 18px",
    "borderRadius": "8px",
    "border": f"1px solid {COLOR['border']}",
    "backgroundColor": COLOR["card"],
    "cursor": "pointer",
    "fontWeight": "600",
    "fontSize": "14px",
    "color": COLOR["text"],
}


_socketio: SocketIO | None = None


def _run_year_sim_job(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    params: dict,
    df_energy_snapshot: pd.DataFrame,
    session_id: str = "",
    ) -> None:
    """Runs the year simulation in a separate thread and emits progress updates via SocketIO."""

    rows: list[dict] = []
    total_days = 0
    try:
        worker_bess = Bess()
        worker_bess.netload_kw = df_energy_snapshot.copy()
        worker_bess.update_battery_params(
            capacity_kwh=params.get("battery_capacity", 30.72),
            max_charge_kw=params.get("battery_max_charge", 8.0),
            max_discharge_kw=params.get("battery_max_discharge", 8.0),
            soc_min_percent=params.get("battery_soc_min", 10.0),
            soc_final_percent=params.get("battery_soc_final", 50.0),
            eta_charge=params.get("battery_eta_charge", 0.936),
            eta_discharge=params.get("battery_eta_discharge", 0.936),
        )

        def _on_progress(done_days: int, total_days_: int, act_day: pd.Timestamp, day_metrics: dict) -> None:
            nonlocal total_days
            total_days = total_days_
            # Compute energy totals from Series before discarding them
            delta_t = 0.25  # 15-min intervals in hours
            p_buy = day_metrics.get("p_buy_kw")
            p_sell = day_metrics.get("p_sell_kw")
            p_ch = day_metrics.get("p_ch_kw")
            grid_import_kwh = float(p_buy.sum() * delta_t) if isinstance(p_buy, pd.Series) else 0.0
            grid_export_kwh = float(p_sell.sum() * delta_t) if isinstance(p_sell, pd.Series) else 0.0
            battery_throughput_kwh = float(p_ch.sum() * delta_t) if isinstance(p_ch, pd.Series) else 0.0
            # Keep only JSON-serialisable (scalar) values
            safe_metrics = {
                k: v for k, v in day_metrics.items()
                if not isinstance(v, (pd.Series, pd.DataFrame))
            }
            if "date" in safe_metrics and safe_metrics["date"] is not None:
                safe_metrics["date"] = str(safe_metrics["date"])
            safe_metrics["grid_import_kwh"] = grid_import_kwh
            safe_metrics["grid_export_kwh"] = grid_export_kwh
            safe_metrics["battery_throughput_kwh"] = battery_throughput_kwh
            rows.append(safe_metrics)
            progress = int(round(100.0 * done_days / total_days_)) if total_days_ > 0 else 100
            state = {
                "status": "running",
                "progress": progress,
                "completed_days": done_days,
                "total_days": total_days_,
                "last_day": str(pd.Timestamp(act_day).date()),
                "error": None,
            }
            if _socketio is not None:
                _socketio.emit("sim_progress", state, room=session_id)

        worker_bess.run_total_simulation(
            start_day=start_ts,
            end_day=end_ts,
            use_dynamic_prices=params.get("use_dynamic_prices", True),
            epex_offset_buy=params.get("epex_offset_buy", 0.0),
            epex_offset_sell=params.get("epex_offset_sell", 0.0),
            grid_fee=params.get("grid_fee", 0.0),
            vat=params.get("vat", 0.0),
            fix_price_buy=params.get("fix_price_buy", 0.0),
            fix_price_sell=params.get("fix_price_sell", 0.0),
            verbose=False,
            progress_callback=_on_progress,
            control_algorithm=params.get("control_algorithm", "model-predictive-control"),
            allow_feed_in=params.get("allow_feed_in", True),
            objective=params.get("objective", "profit"),
        )

        done_state = {
            "status": "done",
            "progress": 100,
            "completed_days": total_days,
            "total_days": total_days,
            "last_day": rows[-1]["date"] if rows else None,
            "rows": rows,
            "error": None,
            "battery_capacity": params.get("battery_capacity", 0.0),
        }
        if _socketio is not None:
            _socketio.emit("sim_progress", done_state, room=session_id)
    except Exception as exc:
        import traceback; traceback.print_exc()
        error_state = {
            "status": "error",
            "progress": 0,
            "completed_days": len(rows),
            "total_days": total_days,
            "last_day": rows[-1]["date"] if rows else None,
            "rows": rows,
            "error": str(exc),
        }
        if _socketio is not None:
            _socketio.emit("sim_progress", error_state, room=session_id)


def _make_error_figure(message: str) -> go.Figure:
    """Returns a blank figure with a centered error message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        height=200,
        margin=dict(t=20, b=20),
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#dc2626"),
            align="center",
        )],
    )
    return fig


def build_figure(lp_results: dict) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Strom Preis (brutto)",
            "Residuallast (ohne Batterieeinfluss)",
            "Batterie State of Charge",
            "Netz Einspeisung / Bezug",
            "Batterie Laden / Entladen",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["act_prices_epex_eur_kwh"].index,
            y=lp_results["act_prices_epex_eur_kwh"].values,
            line_shape="hv",
            name="Epex Preis",
            legendgroup="g1", legend="legend",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["price_sell_eur_kwh"].index,
            y=lp_results["price_sell_eur_kwh"].values,
            line_shape="hv",
            name="Einspeisepreis",
            legendgroup="g1", legend="legend",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["price_buy_eur_kwh"].index,
            y=lp_results["price_buy_eur_kwh"].values,
            line_shape="hv",
            name="Bezugspreise",
            legendgroup="g1", legend="legend",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["net_load_kw"].index,
            y=lp_results["net_load_kw"].values,
            line_shape="hv",
            name="Residuallast",
            legendgroup="g2", legend="legend2",
        ),
        row=2, col=1,
    )

    soc_percent = lp_results["soc_percent"]
    fig.add_trace(
        go.Scatter(
            x=soc_percent.index,
            y=soc_percent.values,
            line_shape="hv",
            name="SOC",
            legendgroup="g3", legend="legend3",
        ),
        row=3, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["p_sell_kw"].index,
            y=lp_results["p_sell_kw"].values,
            line_shape="hv",
            name="Einspeisung",
            legendgroup="g4", legend="legend4",
        ),
        row=4, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["p_buy_kw"].index,
            y=lp_results["p_buy_kw"].values,
            line_shape="hv",
            name="Bezug",
            legendgroup="g4", legend="legend4",
        ),
        row=4, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["p_ch_kw"].index,
            y=lp_results["p_ch_kw"].values,
            line_shape="hv",
            name="Laden",
            legendgroup="g5", legend="legend5",
        ),
        row=5, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lp_results["p_dis_kw"].index,
            y=lp_results["p_dis_kw"].values,
            line_shape="hv",
            name="Entladen",
            legendgroup="g5", legend="legend5",
        ),
        row=5, col=1,
    )

    # Subplot-Domains für Legendenpositionierung auslesen
    y_domains = {}
    for i in range(1, 6):
        ax = f"yaxis{i}" if i > 1 else "yaxis"
        y_domains[i] = fig.layout[ax].domain

    fig.update_layout(
        height=950,
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        margin=dict(t=60, b=30),
        legend=dict(
            yanchor="top", y=y_domains[1][1], xanchor="left", x=1.01,
            font=dict(size=11), tracegroupgap=0,
        ),
        legend2=dict(
            yanchor="top", y=y_domains[2][1], xanchor="left", x=1.01,
            font=dict(size=11), tracegroupgap=0,
        ),
        legend3=dict(
            yanchor="top", y=y_domains[3][1], xanchor="left", x=1.01,
            font=dict(size=11), tracegroupgap=0,
        ),
        legend4=dict(
            yanchor="top", y=y_domains[4][1], xanchor="left", x=1.01,
            font=dict(size=11), tracegroupgap=0,
        ),
        legend5=dict(
            yanchor="top", y=y_domains[5][1], xanchor="left", x=1.01,
            font=dict(size=11), tracegroupgap=0,
        ),
    )

    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=2 * 60 * 60 * 1000,
    )

    fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_yaxes(title_text="Preis [EUR/kWh]", row=1, col=1)
    fig.update_yaxes(title_text="Leistung [kW]", row=2, col=1)
    fig.update_yaxes(title_text="SOC [%]", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="Leistung [kW]", row=4, col=1)
    fig.update_yaxes(title_text="Leistung [kW]", row=5, col=1)

    status = lp_results.get("milp_status", "")
    if status not in ("Optimal", "No battery", "PV surplus"):
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"MILP-Optimierung nicht lösbar: {status}<br>"
                 "Bitte Batterieparameter oder Preismodell prüfen.",
            showarrow=False,
            align="center",
            font=dict(size=13, color="#dc2626"),
            bgcolor="rgba(254,226,226,0.9)",
            bordercolor="#dc2626",
            borderwidth=1,
        )

    try:
        objective_value = lp_results["profit_eur"]
        fig.add_annotation(
            x=1.0,
            y=1.02,
            xref="paper",
            yref="paper",
            text=f"Tagesgewinn: {objective_value:.2f} EUR",
            showarrow=False,
            align="right",
            font=dict(size=12, color="black"),
        )
    except Exception:
        pass

    return fig


def build_year_figure(result_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=["Kumulierter Gewinn", "Gewinn pro Monat", "Monatliche Energiefluesse"],
    )

    if result_df.empty:
        fig.update_layout(
            height=900,
            template="plotly_white",
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#ffffff",
            annotations=[dict(
                text="Noch keine Ergebnisse vorhanden",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=18, color=COLOR["text_light"]),
            )],
        )
        return fig

    df = result_df.copy()
    df = df.sort_index()
    df["cum_profit_eur"] = df["profit_eur"].cumsum()

    monthly = df.resample("MS").agg({
        "profit_eur": "sum",
        "grid_import_kwh": "sum",
        "grid_export_kwh": "sum",
    })

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["cum_profit_eur"],
            mode="lines",
            name="Kumulierter Gewinn",
            line=dict(color="#0f766e", width=3),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in monthly["profit_eur"]]
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["profit_eur"],
            name="Gewinn",
            marker_color=colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["grid_import_kwh"],
            name="Netzbezug",
            marker_color="#2563eb",
            legend="legend",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["grid_export_kwh"],
            name="Netzeinspeisung",
            marker_color="#f59e0b",
            legend="legend",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=900,
        template="plotly_white",
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        barmode="group",
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=70, b=40),
        legend=dict(
            yanchor="top",
            y=0.28,
            xanchor="left",
            x=0.01,
        ),
    )

    fig.update_yaxes(title_text="EUR", row=1, col=1)
    fig.update_yaxes(title_text="EUR", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)

    return fig


def parse_csv(contents: str) -> pd.DataFrame:
    """Parst hochgeladene CSV-Dateien.

    Erkennt automatisch:
    - Semikolon-Format mit "Beginn der Messung;Messwert" (1 Header-Zeile oder VKW mit 4 Metadaten-Zeilen)
    - Generisches CSV (mit/ohne Header, beliebiges Zeitraster)
      → wird bei Bedarf auf 15-Minuten-Intervalle resampelt,
        Watt-Werte (Median > 100) automatisch in kW umgerechnet.
    """
    _, content_string = contents.split(",", maxsplit=1)
    raw = base64.b64decode(content_string)

    try:
        decoded = raw.decode("utf-8-sig")  # utf-8-sig entfernt BOM automatisch
    except UnicodeDecodeError:
        decoded = raw.decode("latin-1")

    lines = decoded.strip().splitlines()
    is_vkw_profile = "Messwert" in decoded and "Beginn der Messung" in decoded

    if is_vkw_profile:
        # Header-Zeile suchen (funktioniert für 1 oder 4 Metadaten-Zeilen)
        header_row = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("Beginn der Messung")),
            None,
        )
        if header_row is None:
            raise ValueError("Keine Zeile mit 'Beginn der Messung' als Header gefunden")
        df = pd.read_csv(StringIO(decoded), sep=";", skiprows=header_row, decimal=",")
        df.columns = df.columns.str.strip()
        df["ts"] = pd.to_datetime(
            df["Beginn der Messung"], format="%d.%m.%Y %H:%M:%S",
        )
        df["ts"] = df["ts"].dt.tz_localize("Europe/Vienna", ambiguous="infer", nonexistent="shift_forward")
        df = df.set_index("ts")
        # Erste Spalte nehmen deren Name mit "Messwert" beginnt (z.B. "Messwert" oder "Messwert-kw")
        val_col = next((c for c in df.columns if c.startswith("Messwert")), None)
        if val_col is None:
            raise ValueError("Keine Messwert-Spalte gefunden")
        df["value_kw"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

    else:
        # Trennzeichen und Header generisch erkennen (unabhängig von Spaltenbezeichnungen)
        import csv as _csv
        sample = decoded[:4096]
        sniffer = _csv.Sniffer()

        # Semikolon-Heuristik: wenn jede Zeile genau ein ";" hat, ist ";"
        # sehr wahrscheinlich das Trennzeichen (und "," nur Dezimalkomma).
        sample_lines = [l for l in lines[:20] if l.strip()]
        if sample_lines and all(l.count(";") == 1 for l in sample_lines):
            sep = ";"
        else:
            try:
                dialect = sniffer.sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except _csv.Error:
                sep = ","
        decimal = "," if sep != "," else "."

        # Robustere Header-Erkennung: sniffer.has_header ist unzuverlässig
        # bei rein numerischen CSVs ohne echten Header.  Wir prüfen, ob die
        # erste Zelle der ersten Zeile ein Datum sein könnte – wenn ja, gibt
        # es keinen Header.
        has_header = sniffer.has_header(sample)
        if has_header:
            first_line = lines[0].split(sep)[0].strip()
            try:
                pd.to_datetime(first_line, dayfirst=True)
                has_header = False          # erste Zeile ist ein Datenpunkt
            except (ValueError, TypeError):
                pass                        # echte Header-Zeile

        df = pd.read_csv(
            StringIO(decoded), sep=sep, decimal=decimal,
            header=0 if has_header else None,
        )
        df.columns = [str(c).strip() for c in df.columns]

        if not has_header:
            ts_col, val_col = df.columns[0], df.columns[1]
        else:
            ts_col = None
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].head(5))
                    ts_col = col
                    break
                except (ValueError, TypeError):
                    continue
            if ts_col is None:
                raise ValueError("Keine Timestamp-Spalte erkannt")
            val_col = None
            for col in df.columns:
                if col == ts_col:
                    continue
                if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.5:
                    val_col = col
                    break
            if val_col is None:
                raise ValueError("Keine numerische Wert-Spalte erkannt")

        # Zeitstempel parsen: deutsches Format bevorzugt, UTC nur wenn
        # die Timestamps tatsächlich Zeitzonen-Info enthalten.
        _ts_parsed = False
        for _fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                df["ts"] = pd.to_datetime(
                    df[ts_col], format=_fmt, dayfirst=True,
                ).dt.tz_localize("Europe/Vienna", ambiguous="infer", nonexistent="shift_forward")
                _ts_parsed = True
                break
            except (ValueError, TypeError):
                continue
        if not _ts_parsed:
            try:
                df["ts"] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert("Europe/Vienna")
            except Exception:
                df["ts"] = pd.to_datetime(
                    df[ts_col], dayfirst=True,
                ).dt.tz_localize("Europe/Vienna", ambiguous="infer", nonexistent="shift_forward")
        df = df.set_index("ts")
        df["value_kw"] = pd.to_numeric(
            df[val_col], errors="coerce").fillna(0.0)

        # Auf 15-min resampeln falls nötig
        median_freq = df.index.to_series().diff().dropna().median()
        if median_freq != pd.Timedelta(minutes=15):
            df = df[["value_kw"]].resample("15min").mean().fillna(0.0)

        # Watt → kW falls Werte zu groß für kW
        if df["value_kw"].abs().quantile(0.75) > 100:
            df["value_kw"] = df["value_kw"] / 1000.0

    # Keine Zukunfts-Timestamps (heutiger Tag ist noch unvollständig)
    now = pd.Timestamp.now(tz="Europe/Vienna")
    df = df[df.index < now.normalize()]

    return df[["value_kw"]]


def run_dashboard(
    bess: Bess,
    use_dynamic_prices: bool = True,
    port: int = 8051,
    debug: bool = False,
    ) -> Dash:

    global _socketio

    app = Dash(
        __name__,
        title="FHV FZE - Lastmanagement Simulation",
        external_scripts=[
            "https://cdn.socket.io/4.7.5/socket.io.min.js",
        ],
    )
    app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <!-- Matomo -->
        <script>
            var _paq = window._paq = window._paq || [];
            _paq.push(['trackPageView']);
            _paq.push(['enableLinkTracking']);
            (function() {
                var u="//webstats.it.fhv.at/";
                _paq.push(['setTrackerUrl', u+'matomo.php']);
                _paq.push(['setSiteId', '1']);
                var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
                g.async=true; g.src=u+'matomo.js'; s.parentNode.insertBefore(g,s);
            })();
        </script>
        <!-- End Matomo Code -->
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''
    _socketio = SocketIO(app.server, async_mode="gevent", cors_allowed_origins="*")

    @_socketio.on("join")
    def _on_join(data):
        join_room(data.get("session_id", ""))

    # Verfügbare Tage aus dem aktuellen Datensatz
    netload_kw = bess.get_netload_profile()
    _first_ts = netload_kw.index.min()
    _first_cal = _first_ts.date()
    if _first_ts.hour != 0 or _first_ts.minute != 0:
        _first_cal = _first_cal + pd.DateOffset(days=1)
    _last_ts = netload_kw.index.max()
    _last_cal = _last_ts.date()
    if _last_ts.hour != 23 or _last_ts.minute != 45:
        _last_cal = _last_cal - pd.DateOffset(days=1)
    _first_date = pd.Timestamp(_first_cal, tz="Europe/Vienna")
    _last_date = pd.Timestamp(_last_cal, tz="Europe/Vienna")

    # ── Layout (Funktion → jeder Page-Load bekommt eigene Session-ID) ─
    def _serve_layout():
        return html.Div(
            style={"backgroundColor": COLOR["bg"], "minHeight": "100vh",
                    "fontFamily": "Inter, system-ui, -apple-system, sans-serif"},
            children=[
                dcc.Store(id="session-id", data=str(uuid.uuid4())),
                dcc.Store(id="ws-sim-progress"),
                dcc.Store(id="socket-init"),
            # ── Header ───────────────────────────────────────────────────
            html.Div(
                style={"backgroundColor": COLOR["header"],
                       "padding": "18px 40px",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.12)",
                       "display": "flex",
                       "alignItems": "center",
                       "justifyContent": "space-between"},
                children=[
                    html.H1(
                        "FHV FZE - Lastmanagement Simulation",
                        style={"color": "#fff", "margin": "0",
                               "fontSize": "22px", "fontWeight": "700",
                               "letterSpacing": "-0.3px"},
                    ),
                    html.A(
                        # GitHub SVG icon (Octicon mark-github)
                        [
                            html.Img(
                                src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg",
                                style={"height": "24px", "width": "24px",
                                       "filter": "invert(1)"},
                            ),
                            html.Span("Open Source", style={
                                "color": "#fff", "fontSize": "13px",
                                "marginLeft": "6px", "fontWeight": "500",
                            }),
                        ],
                        href="https://github.com/erc-fhv/Home_BESS",
                        target="_blank",
                        title="GitHub Repository",
                        style={"display": "flex", "alignItems": "center",
                               "gap": "2px", "textDecoration": "none",
                               "opacity": "0.8", "transition": "opacity 0.2s"},
                    ),
                ],
            ),

            # ── Disclaimer-Banner ─────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": "#fefce8",
                    "borderBottom": "1px solid #fde047",
                    "padding": "10px 40px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                },
                children=[
                    html.Span("⚠️", style={"fontSize": "16px"}),
                    html.Span(
                        [
                            html.Strong("Entwicklungs-Tool - keine Haftung: "),
                            "Dieses Tool wurde im Rahmen eines Forschungsprojekts an der FH Vorarlberg nach bestem Wissen und Gewissen entwickelt. "
                            "Die Simulationsergebnisse dienen ausschließlich der Orientierung und erheben keinen Anspruch auf Vollständigkeit oder Richtigkeit. "
                            "Für wirtschaftliche Entscheidungen, die auf Basis dieser Ergebnisse getroffen werden, wird keine Haftung übernommen.",
                        ],
                        style={"fontSize": "13px", "color": "#713f12", "lineHeight": "1.5"},
                    ),
                ],
            ),

            # ── Main: Sidebar left + Graph right ─────────────────────────
            html.Div(
                style={"display": "flex", "gap": "20px",
                       "maxWidth": "1600px", "margin": "24px auto",
                       "padding": "0 24px", "alignItems": "flex-start"},
                children=[
                    # ── Left sidebar (controls) ──────────────────────────
                    html.Div(
                        style={"flex": "0 0 280px", "display": "flex",
                               "flexDirection": "column", "gap": "14px"},
                        children=[
                            html.Div(style=CARD, children=[
                                html.Div("Daten Input", style=CARD_TITLE),
                                dcc.RadioItems(
                                    id="input-mode",
                                    options=[
                                        {"label": " Residuallast",
                                         "value": "residual"},
                                        {"label": " Last und Erzeugung",
                                         "value": "load_gen"},
                                    ],
                                    value="residual",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "6px",
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                                html.Hr(style={
                                    "border": "none",
                                    "borderTop": f"1px solid {COLOR['border']}",
                                    "margin": "10px 0"}),
                                html.Div(
                                    id="input-selected-profile",
                                    style={
                                        "fontSize": "12px",
                                        "color": COLOR["text_light"],
                                        "backgroundColor": "#f8fafc",
                                        "borderRadius": "6px",
                                        "padding": "10px 12px",
                                        "lineHeight": "1.5",
                                        "marginBottom": "10px",
                                        "border": f"1px solid {COLOR['border']}",
                                    },
                                    children=[
                                        html.Span(
                                            ("Aktuell verwendetes Lastprofil: Realer Beispiel Haushalt mit PV.",),
                                            style={"fontWeight": "600"})
                                    ],
                                ),
                                html.Div(
                                    id="input-instructions",
                                    style={
                                        "fontSize": "12px",
                                        "color": COLOR["text_light"],
                                        "backgroundColor": "#f8fafc",
                                        "borderRadius": "6px",
                                        "padding": "10px 12px",
                                        "lineHeight": "1.5",
                                        "marginBottom": "10px",
                                        "border": f"1px solid {COLOR['border']}",
                                    },
                                    children=[
                                        html.Span(
                                            ("Optional neues Profil (ohne Batterieeinfluss) als CSV hochladen. ",
                                             "Entweder ein VKW-Online-Service Export. ",
                                             "Oder ein eigenes Profil mit erster Spalte Datum/Uhrzeit und zweite Spalte Leistungswerte in kW."),
                                            style={"fontWeight": "600"})
                                    ],
                                ),
                                # Upload: Residuallast (1 CSV)
                                html.Div(
                                    id="upload-residual",
                                    children=[
                                        dcc.Upload(
                                            id="residual-profile-upload",
                                            children=html.Button(
                                                "Upload Residuallast",
                                                style={**BTN,
                                                       "width": "100%"}),
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="residual-upload-status",
                                            style={"marginTop": "6px", "fontSize": "12px"},
                                        ),
                                    ],
                                ),
                                # Upload: Last + Erzeugung (2 CSVs)
                                html.Div(
                                    id="upload-load-gen",
                                    style={"display": "none"},
                                    children=[
                                        dcc.Upload(
                                            id="load-profile-upload",
                                            children=html.Button(
                                                "Upload Last",
                                                style={**BTN,
                                                       "width": "100%"}),
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="load-upload-status",
                                            style={"marginTop": "6px", "fontSize": "12px"},
                                        ),
                                        html.Div(style={"height": "6px"}),
                                        dcc.Upload(
                                            id="gen-profile-upload",
                                            children=html.Button(
                                                "Upload PV",
                                                style={**BTN,
                                                       "width": "100%"}),
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="gen-upload-status",
                                            style={"marginTop": "6px", "fontSize": "12px"},
                                        ),
                                    ],
                                ),
                            ]),

                            # Card: Steuerung
                            html.Div(style=CARD, children=[
                                html.Div("Steuerung",
                                         style=CARD_TITLE),
                                dcc.RadioItems(
                                    id="control-algorithm",
                                    options=[
                                        {"label": " PV-\u00dcberschussladen",
                                         "value": "pv-ueberschussladen",
                                         "disabled": False},
                                        {"label": " Mathematische Optimierung (MILP)",
                                         "value": "model-predictive-control"},
                                        {"label": " Ohne Batterie",
                                         "value": "no-control"},
                                    ],
                                    value="model-predictive-control",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "6px",
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                                html.Hr(style={
                                    "border": "none",
                                    "borderTop": f"1px solid {COLOR['border']}",
                                    "margin": "12px 0"}),
                                dcc.Checklist(
                                    id="allow-feed-in",
                                    options=[{
                                        "label": " Batterie-Einspeisung vermeiden",
                                        "value": "yes",
                                    }],
                                    value=[],
                                    labelStyle={
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                                dcc.RadioItems(
                                    id="opt-objective",
                                    options=[
                                        {"label": " Profit maximieren",
                                         "value": "profit"},
                                        {"label": " Autarkie maximieren",
                                         "value": "autarky"},
                                        {"label": " Netzspitzen vermeiden",
                                         "value": "peak_shaving"},
                                    ],
                                    value="profit",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "6px",
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                            ]),

                            # Card: Preismodell
                            html.Div(style=CARD, children=[
                                html.Div("Preismodell",
                                         style=CARD_TITLE),
                                dcc.RadioItems(
                                    id="price-source",
                                    options=[
                                        {"label": " EPEX Day-Ahead",
                                         "value": "epex"},
                                        {"label": " Fixpreis",
                                         "value": "fix",
                                         "disabled": False},
                                    ],
                                    value="epex",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "5px",
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                                # EPEX Offsets
                                html.Div(
                                    id="epex-params",
                                    style={"marginTop": "10px"},
                                    children=[
                                        html.Div("Einspeisung Abschlag [ct/kWh]",
                                                 style={"fontSize": "12px",
                                                        "color": COLOR["text_light"],
                                                        "marginBottom": "3px"}),
                                        dcc.Input(
                                            id="epex-offset-sell",
                                            type="number",
                                            value=0.6, step=0.1,
                                            style={"width": "100%",
                                                   "padding": "6px 10px",
                                                   "borderRadius": "6px",
                                                   "border": f"1px solid {COLOR['border']}",
                                                   "fontSize": "13px"},
                                        ),
                                        html.Div("Bezug Aufschlag [ct/kWh]",
                                                 style={"fontSize": "12px",
                                                        "color": COLOR["text_light"],
                                                        "marginBottom": "3px"}),
                                        dcc.Input(
                                            id="epex-offset-buy",
                                            type="number",
                                            value=1.44, step=0.01,
                                            style={"width": "100%",
                                                   "padding": "6px 10px",
                                                   "borderRadius": "6px",
                                                   "border": f"1px solid {COLOR['border']}",
                                                   "fontSize": "13px",
                                                   "marginBottom": "8px"},
                                        ),
                                    ],
                                ),
                                # Fixpreise
                                html.Div(
                                    id="fix-params",
                                    style={"display": "none",
                                           "marginTop": "10px"},
                                    children=[
                                        html.Div("Bezugspreis [ct/kWh]",
                                                 style={"fontSize": "12px",
                                                        "color": COLOR["text_light"],
                                                        "marginBottom": "3px"}),
                                        dcc.Input(
                                            id="fix-price-buy",
                                            type="number",
                                            value=12.72, step=0.01, min=0,
                                            style={"width": "100%",
                                                   "padding": "6px 10px",
                                                   "borderRadius": "6px",
                                                   "border": f"1px solid {COLOR['border']}",
                                                   "fontSize": "13px",
                                                   "marginBottom": "8px"},
                                        ),
                                        html.Div("Einspeisepreis [ct/kWh]",
                                                 style={"fontSize": "12px",
                                                        "color": COLOR["text_light"],
                                                        "marginBottom": "3px"}),
                                        dcc.Input(
                                            id="fix-price-sell",
                                            type="number",
                                            value=9.0, step=0.1, min=0,
                                            style={"width": "100%",
                                                   "padding": "6px 10px",
                                                   "borderRadius": "6px",
                                                   "border": f"1px solid {COLOR['border']}",
                                                   "fontSize": "13px"},
                                        ),
                                    ],
                                ),
                                # Netzentgelt + USt (gilt für beide Preismodelle)
                                html.Hr(style={
                                    "border": "none",
                                    "borderTop": f"1px solid {COLOR['border']}",
                                    "margin": "10px 0"}),
                                html.Div("Bezug Netzentgelt [ct/kWh]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="grid-fee",
                                    type="number",
                                    value=6.0, step=0.1, min=0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Bezug Umsatzsteuer [%]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="vat",
                                    type="number",
                                    value=20.0, step=0.1, min=0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px"},
                                ),
                            ]),

                            # Card: Batterie-Einstellungen
                            html.Div(style=CARD, children=[
                                html.Div("Batterie-Einstellungen",
                                         style=CARD_TITLE),
                                html.Div("Kapazität [kWh]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-capacity",
                                    type="number",
                                    value=30.72, step=0.01, min=0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Max. Ladeleistung [kW]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-max-charge",
                                    type="number",
                                    value=8.0, step=0.1, min=0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Max. Entladeleistung [kW]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-max-discharge",
                                    type="number",
                                    value=8.0, step=0.1, min=0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Min. SOC [%]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-soc-min",
                                    type="number",
                                    value=10.0, step=1.0, min=0, max=100,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Ziel-SOC Ende [%]",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-soc-final",
                                    type="number",
                                    value=50.0, step=1.0, min=0, max=100,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Lade-Wirkungsgrad",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-eta-charge",
                                    type="number",
                                    value=0.936, step=0.001, min=0.5, max=1.0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px",
                                           "marginBottom": "8px"},
                                ),
                                html.Div("Entlade-Wirkungsgrad",
                                         style={"fontSize": "12px",
                                                "color": COLOR["text_light"],
                                                "marginBottom": "3px"}),
                                dcc.Input(
                                    id="battery-eta-discharge",
                                    type="number",
                                    value=0.936, step=0.001, min=0.5, max=1.0,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px"},
                                ),
                            ]),
                        ],
                    ),

                    # ── Right side (tabs) ─────────────────────────────
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=dcc.Tabs(
                            id="main-tabs",
                            value="tab-day",
                            style={"marginBottom": "14px"},
                            children=[
                                # ── Tab 1: Tagesansicht ──────────────
                                dcc.Tab(
                                    label="Tagesansicht",
                                    value="tab-day",
                                    style={"padding": "10px 20px",
                                           "fontWeight": "500"},
                                    selected_style={
                                        "padding": "10px 20px",
                                        "fontWeight": "700",
                                        "borderTop": f"3px solid {COLOR['accent']}",
                                    },
                                    children=[
                                        # Datumsnavigation
                                        html.Div(
                                            style={"display": "flex",
                                                   "alignItems": "center",
                                                   "gap": "8px",
                                                   "margin": "14px 0"},
                                            children=[
                                                html.Button(
                                                    "\u25c0", id="prev-day",
                                                    n_clicks=0,
                                                    style={**BTN,
                                                           "fontSize": "18px",
                                                           "padding": "6px 14px"}),
                                                dcc.DatePickerSingle(
                                                    id="act-day-picker",
                                                    min_date_allowed="2025-01-01",
                                                    max_date_allowed="2025-12-31",
                                                    date="2025-11-05",
                                                    display_format="DD.MM.YYYY",
                                                ),
                                                html.Button(
                                                    "\u25b6", id="next-day",
                                                    n_clicks=0,
                                                    style={**BTN,
                                                           "fontSize": "18px",
                                                           "padding": "6px 14px"}),
                                            ],
                                        ),
                                        # Graph
                                        html.Div(
                                            style={**CARD,
                                                   "padding": "12px 16px"},
                                            children=dcc.Graph(
                                                id="result-graph",
                                                style={"height": "950px"}),
                                        ),
                                    ],
                                ),

                                # ── Tab 2: Gesamtsimulation ──────────
                                dcc.Tab(
                                    label="Gesamtsimulation",
                                    value="tab-year",
                                    style={"padding": "10px 20px",
                                           "fontWeight": "500"},
                                    selected_style={
                                        "padding": "10px 20px",
                                        "fontWeight": "700",
                                        "borderTop": f"3px solid {COLOR['accent']}",
                                    },
                                    children=[
                                        html.Div(
                                            style={**CARD,
                                                   "marginTop": "14px",
                                                   "padding": "24px"},
                                            children=[
                                                html.Div(
                                                    style={"display": "flex", "gap": "10px", "alignItems": "center",
                                                           "flexWrap": "wrap", "marginBottom": "12px"},
                                                    children=[
                                                        dcc.DatePickerRange(
                                                            id="year-range-picker",
                                                            min_date_allowed=_first_date.date(),
                                                            max_date_allowed=_last_date.date(),
                                                            start_date=_first_date.date(),
                                                            end_date=_last_date.date(),
                                                            display_format="DD.MM.YYYY",
                                                        ),
                                                        html.Button(
                                                            "Starte Gesamtsimulation",
                                                            id="start-year-sim",
                                                            n_clicks=0,
                                                            style={**BTN, "backgroundColor": COLOR["accent"], "color": "#fff"},
                                                        ),
                                                    ],
                                                ),
                                                html.Progress(
                                                    id="year-progress",
                                                    value="0",
                                                    max=100,
                                                    style={"width": "100%", "height": "16px", "marginBottom": "6px"},
                                                ),
                                                html.Div(
                                                    "Bereit.",
                                                    id="year-progress-text",
                                                    style={"fontSize": "13px", "color": COLOR["text_light"], "marginBottom": "14px"},
                                                ),
                                                html.Div(
                                                    id="year-summary",
                                                    style={
                                                        "fontSize": "14px",
                                                        "fontWeight": "600",
                                                        "color": COLOR["text"],
                                                        "marginBottom": "12px",
                                                    },
                                                ),
                                                dcc.Graph(
                                                    id="year-graph",
                                                    figure=build_year_figure(pd.DataFrame()),
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )

    app.layout = _serve_layout

    # ── Server-seitige Caches (per Session, LRU-begrenzt) ────────────
    _pv_caches: _BoundedCache = _BoundedCache()
    _netload_cache: _BoundedCache = _BoundedCache()

    # ── Callbacks ────────────────────────────────────────────────────────
    @app.callback(
        Output("upload-residual", "style"),
        Output("upload-load-gen", "style"),
        Input("input-mode", "value"),
    )
    def toggle_upload_mode(mode):
        if mode == "load_gen":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("epex-params", "style"),
        Output("fix-params", "style"),
        Input("price-source", "value"),
    )
    def toggle_price_params(source):
        if source == "fix":
            return {"display": "none"}, {"display": "block", "marginTop": "10px"}
        return {"display": "block", "marginTop": "10px"}, {"display": "none"}

    @app.callback(
        Output("allow-feed-in", "options"),
        Output("opt-objective", "options"),
        Output("battery-capacity", "disabled"),
        Output("battery-max-charge", "disabled"),
        Output("battery-max-discharge", "disabled"),
        Output("battery-soc-min", "disabled"),
        Output("battery-soc-final", "disabled"),
        Output("battery-eta-charge", "disabled"),
        Output("battery-eta-discharge", "disabled"),
        Input("control-algorithm", "value"),
    )
    def toggle_mpc_options(algorithm):
        disabled = algorithm != "model-predictive-control"
        no_battery = algorithm == "no-control"
        feed_in_opts = [{
            "label": " Batterie-Einspeisung vermeiden",
            "value": "yes",
            "disabled": disabled,
        }]
        obj_opts = [
            {"label": " Maximiere Profit",
             "value": "profit", "disabled": disabled},
            {"label": " Maximiere Autarkie",
             "value": "autarky", "disabled": disabled},
            {"label": " Minimiere Netzspitzen",
             "value": "peak_shaving", "disabled": disabled},
        ]
        return (feed_in_opts, obj_opts,
                no_battery, no_battery, no_battery, no_battery,
                no_battery, no_battery, no_battery)

    @app.callback(
        Output("act-day-picker", "date"),
        Input("prev-day", "n_clicks"),
        Input("next-day", "n_clicks"),
        State("act-day-picker", "date"),
        State("act-day-picker", "min_date_allowed"),
        State("act-day-picker", "max_date_allowed"),
    )
    def shift_day(prev_clicks, next_clicks, current_date, min_date, max_date):
        if current_date is None:
            return current_date

        ctx = callback_context
        if not ctx.triggered:
            return current_date

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        date = pd.Timestamp(current_date)

        if triggered_id == "prev-day":
            date -= pd.Timedelta(days=1)
        elif triggered_id == "next-day":
            date += pd.Timedelta(days=1)

        if min_date:
            date = max(pd.Timestamp(min_date), date)
        if max_date:
            date = min(pd.Timestamp(max_date), date)

        return date.date()

    @app.callback(
        Output("result-graph", "figure"),
        Output("act-day-picker", "date", allow_duplicate=True),
        Output("act-day-picker", "min_date_allowed"),
        Output("act-day-picker", "max_date_allowed"),
        Output("year-range-picker", "min_date_allowed"),
        Output("year-range-picker", "max_date_allowed"),
        Output("year-range-picker", "start_date"),
        Output("year-range-picker", "end_date"),
        Output("input-selected-profile", "children"),
        Output("residual-upload-status", "children"),
        Output("load-upload-status", "children"),
        Output("gen-upload-status", "children"),
        Input("act-day-picker", "date"),
        Input("residual-profile-upload", "contents"),
        Input("load-profile-upload", "contents"),
        Input("gen-profile-upload", "contents"),
        Input("price-source", "value"),
        Input("epex-offset-buy", "value"),
        Input("epex-offset-sell", "value"),
        Input("grid-fee", "value"),
        Input("vat", "value"),
        Input("fix-price-buy", "value"),
        Input("fix-price-sell", "value"),
        Input("battery-capacity", "value"),
        Input("battery-max-charge", "value"),
        Input("battery-max-discharge", "value"),
        Input("battery-soc-min", "value"),
        Input("battery-soc-final", "value"),
        Input("battery-eta-charge", "value"),
        Input("battery-eta-discharge", "value"),
        Input("control-algorithm", "value"),
        Input("allow-feed-in", "value"),
        Input("opt-objective", "value"),
        State("input-mode", "value"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def update_graph(act_day, residual_contents, load_contents,
                     gen_contents, price_source, epex_offset_buy_cent,
                     epex_offset_sell_cent, grid_fee_cent, vat,
                     fix_price_buy_cent, fix_price_sell_cent,
                     battery_capacity, battery_max_charge, battery_max_discharge,
                     battery_soc_min, battery_soc_final, battery_eta_charge, battery_eta_discharge,
                     control_algorithm, allow_feed_in_val, opt_objective,
                     input_mode, session_id):

        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        _sid = session_id or ""

        profile_label_out = no_update
        status_residual_out = no_update
        status_load_out = no_update
        status_gen_out = no_update

        _ok_style = {"color": "#16a34a", "fontWeight": "600"}
        _err_style = {"color": "#dc2626", "fontWeight": "600"}

        # Nur bei Upload neu parsen, nicht bei Datumswechsel
        if triggered in ("residual-profile-upload", "load-profile-upload", "gen-profile-upload"):
            if input_mode == "load_gen":
                df_load = None
                df_gen = None
                if load_contents:
                    try:
                        df_load = parse_csv(load_contents)
                        status_load_out = [html.Span(f"✓ {len(df_load)} Messzeitpunkte geladen", style=_ok_style)]
                    except Exception as exc:
                        status_load_out = [html.Span(f"✗ {exc}", style=_err_style)]
                if gen_contents:
                    try:
                        df_gen = parse_csv(gen_contents)
                        status_gen_out = [html.Span(f"✓ {len(df_gen)} Messzeitpunkte geladen", style=_ok_style)]
                    except Exception as exc:
                        status_gen_out = [html.Span(f"✗ {exc}", style=_err_style)]
                if df_load is not None and df_gen is not None:
                    net = (df_load["value_kw"] - df_gen["value_kw"]).fillna(0.0)
                    _netload_cache[_sid] = pd.DataFrame({"net_load_kw": net})
                    profile_label_out = [html.Span("Aktuell verwendetes Lastprofil: Eigenes Profil",
                                                   style={"fontWeight": "600"})]
                else:
                    if df_load is None and df_gen is None:
                        msg = "Kein Profil konnte geladen werden."
                    elif df_load is None:
                        msg = "Bitte noch Last-Profil hochladen."
                    else:
                        msg = "Bitte noch PV-Profil hochladen."
                    profile_label_out = [html.Span(msg, style={"fontWeight": "600",
                                                               "color": COLOR["text_light"]})]
                    return (
                        no_update, no_update, no_update, no_update,
                        no_update, no_update, no_update, no_update,
                        profile_label_out,
                        status_residual_out, status_load_out, status_gen_out,
                    )
            elif residual_contents:
                try:
                    df_res = parse_csv(residual_contents)
                    _netload_cache[_sid] = pd.DataFrame({"net_load_kw": df_res["value_kw"]})
                    status_residual_out = [html.Span(f"✓ {len(df_res)} Messzeitpunkte geladen", style=_ok_style)]
                    profile_label_out = [html.Span("Aktuell verwendetes Lastprofil: Eigenes Profil",
                                                   style={"fontWeight": "600"})]
                except Exception as exc:
                    status_residual_out = [html.Span(f"✗ {exc}", style=_err_style)]
                    return (
                        _make_error_figure(f"CSV konnte nicht eingelesen werden:<br>{exc}"),
                        no_update, no_update, no_update, no_update,
                        no_update, no_update, no_update, no_update,
                        status_residual_out, status_load_out, status_gen_out,
                    )

        # Worker-Bess (per-Request, shared EPEX prices)
        worker = copy.copy(bess)
        _cached_netload = _netload_cache.get(_sid)
        if _cached_netload is not None:
            worker.netload_kw = _cached_netload

        netload_profile = worker.get_netload_profile()
        first_ts = netload_profile.index.min()
        min_cal = first_ts.date()
        if first_ts.hour != 0 or first_ts.minute != 0:
            min_cal = min_cal + pd.DateOffset(days=1)
        last_ts = netload_profile.index.max()
        max_cal = last_ts.date()
        if last_ts.hour != 23 or last_ts.minute != 45:
            max_cal = max_cal - pd.DateOffset(days=1)
        min_date = pd.Timestamp(min_cal, tz="Europe/Vienna")
        max_date = pd.Timestamp(max_cal, tz="Europe/Vienna")

        if max_date < min_date:
            return (
                _make_error_figure(
                    "Keine vollständigen Tage im Datensatz verfügbar.<br>"
                    "Bitte eine CSV mit mindestens einem vollständigen Tag (00:00–23:45) hochladen."
                ),
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                status_residual_out, status_load_out, status_gen_out,
            )

        act_day = pd.Timestamp(act_day, tz="Europe/Vienna")
        # Bei neuem Upload auf ersten verfuegbaren Tag springen
        is_upload = triggered in ("residual-profile-upload", "load-profile-upload", "gen-profile-upload")
        if is_upload:
            act_day = min_date if min_date.tzinfo else min_date.tz_localize("Europe/Vienna")
        # Sicherstellen, dass act_day im verfügbaren Bereich liegt
        act_day = max(act_day, min_date)
        act_day = min(act_day, max_date)

        # year-range-picker: Grenzen immer aktualisieren, Auswahl nur bei Upload
        year_start_out = min_date.date() if is_upload else no_update
        year_end_out = max_date.date() if is_upload else no_update

        # Batterie-Parameter aktualisieren
        worker.update_battery_params(
            capacity_kwh=battery_capacity,
            max_charge_kw=battery_max_charge,
            max_discharge_kw=battery_max_discharge,
            soc_min_percent=battery_soc_min,
            soc_final_percent=battery_soc_final,
            eta_charge=battery_eta_charge,
            eta_discharge=battery_eta_discharge,
        )

        epex_offset_buy_eur = (epex_offset_buy_cent or 0) / 100.0
        epex_offset_sell_eur = (epex_offset_sell_cent or 0) / 100.0
        grid_fee_eur = (grid_fee_cent or 0) / 100.0
        vat_fraction = (vat or 0) / 100.0
        fix_price_buy_eur = (fix_price_buy_cent or 0) / 100.0
        fix_price_sell_eur = (fix_price_sell_cent or 0) / 100.0

        sim_kwargs = dict(
            use_dynamic_prices=(price_source == "epex"),
            epex_offset_buy_eur_kwh=epex_offset_buy_eur,
            epex_offset_sell_eur_kwh=epex_offset_sell_eur,
            grid_fee_eur_kwh=grid_fee_eur,
            vat=vat_fraction,
            fix_price_buy_eur_kwh=fix_price_buy_eur,
            fix_price_sell_eur_kwh=fix_price_sell_eur,
            verbose=False,
            allow_feed_in="yes" not in (allow_feed_in_val or []),
            objective=opt_objective or "profit",
        )

        try:
            if control_algorithm == "pv-ueberschussladen":
                # Build cache key from all parameters that affect PV surplus results
                _pv_cache = _pv_caches.get_or_create(session_id or "")
                profile_fp = (
                    str(netload_profile.index.min()),
                    str(netload_profile.index.max()),
                    len(netload_profile),
                )
                params_key = (
                    price_source, epex_offset_buy_cent, epex_offset_sell_cent,
                    grid_fee_cent, vat, fix_price_buy_cent, fix_price_sell_cent,
                    battery_capacity, battery_max_charge, battery_max_discharge,
                    battery_soc_min, battery_soc_final, battery_eta_charge,
                    battery_eta_discharge, profile_fp,
                )
                if _pv_cache.get("params_key") != params_key:
                    # Recompute full dataset with SOC carry-over
                    _pv_cache.clear()
                    _pv_cache["params_key"] = params_key
                    soc = battery_soc_final or 0.0
                    all_days = pd.date_range(
                        start=min_date, end=max_date, freq="1D", tz="Europe/Vienna",
                    )
                    for day in all_days:
                        day_res = worker.run(
                            act_day=day, **sim_kwargs,
                            control_algorithm="pv-ueberschussladen",
                            soc_init_percent=soc,
                        )
                        _pv_cache[str(day.date())] = day_res
                        soc_series = day_res.get("soc_percent")
                        if isinstance(soc_series, pd.Series) and len(soc_series) > 0:
                            soc = float(soc_series.iloc[-1])

                lp_results = _pv_cache.get(str(act_day.date()))
                if lp_results is None:
                    lp_results = worker.run(
                        act_day=act_day, **sim_kwargs,
                        control_algorithm="pv-ueberschussladen",
                    )
            else:
                lp_results = worker.run(
                    act_day=act_day, **sim_kwargs,
                    control_algorithm=control_algorithm,
                )
        except Exception as exc:
            return (
                _make_error_figure(f"Simulation fehlgeschlagen:<br>{exc}"),
                act_day.date(), min_date.date(), max_date.date(),
                min_date.date(), max_date.date(), year_start_out, year_end_out,
                no_update,
                status_residual_out, status_load_out, status_gen_out,
            )

        status = lp_results.get("milp_status", "")
        if status not in ("Optimal", "No battery", "PV surplus"):
            return (
                _make_error_figure(
                    f"MILP-Optimierung nicht lösbar: <b>{status}</b><br>"
                    "Bitte Batterieparameter (SOC-Grenzen, Kapazität) oder Preismodell prüfen."
                ),
                act_day.date(), min_date.date(), max_date.date(),
                min_date.date(), max_date.date(), year_start_out, year_end_out,
                no_update,
                no_update, no_update, no_update,
            )

        return (
            build_figure(lp_results), act_day.date(), min_date.date(), max_date.date(),
            min_date.date(), max_date.date(), year_start_out, year_end_out,
            profile_label_out,
            status_residual_out, status_load_out, status_gen_out,
        )

    # ── Start-Callback: Gesamtsimulation starten ────────────────────
    @app.callback(
        Output("year-progress", "value", allow_duplicate=True),
        Output("year-progress-text", "children", allow_duplicate=True),
        Output("year-summary", "children", allow_duplicate=True),
        Output("year-graph", "figure", allow_duplicate=True),
        Input("start-year-sim", "n_clicks"),
        State("year-range-picker", "start_date"),
        State("year-range-picker", "end_date"),
        State("price-source", "value"),
        State("epex-offset-buy", "value"),
        State("epex-offset-sell", "value"),
        State("grid-fee", "value"),
        State("vat", "value"),
        State("fix-price-buy", "value"),
        State("fix-price-sell", "value"),
        State("battery-capacity", "value"),
        State("battery-max-charge", "value"),
        State("battery-max-discharge", "value"),
        State("battery-soc-min", "value"),
        State("battery-soc-final", "value"),
        State("battery-eta-charge", "value"),
        State("battery-eta-discharge", "value"),
        State("control-algorithm", "value"),
        State("allow-feed-in", "value"),
        State("opt-objective", "value"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def start_total_simulation(
        n_clicks,
        start_date,
        end_date,
        price_source,
        epex_offset_buy,
        epex_offset_sell,
        grid_fee,
        vat,
        fix_price_buy,
        fix_price_sell,
        battery_capacity,
        battery_max_charge,
        battery_max_discharge,
        battery_soc_min,
        battery_soc_final,
        battery_eta_charge,
        battery_eta_discharge,
        control_algorithm,
        allow_feed_in_val,
        opt_objective,
        session_id,
    ):
        if not n_clicks or not start_date or not end_date:
            raise PreventUpdate

        start_ts = pd.Timestamp(start_date, tz="Europe/Vienna").normalize()
        end_ts = pd.Timestamp(end_date, tz="Europe/Vienna").normalize()
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        days = pd.date_range(start=start_ts, end=end_ts, freq="1D", tz="Europe/Vienna")

        epex_offset_buy_eur = (epex_offset_buy or 0) / 100.0
        epex_offset_sell_eur = (epex_offset_sell or 0) / 100.0
        grid_fee_eur = (grid_fee or 0) / 100.0
        vat_fraction = (vat or 0) / 100.0
        fix_price_buy_eur = (fix_price_buy or 0) / 100.0
        fix_price_sell_eur = (fix_price_sell or 0) / 100.0

        params = {
            "use_dynamic_prices": price_source == "epex",
            "epex_offset_buy": epex_offset_buy_eur,
            "epex_offset_sell": epex_offset_sell_eur,
            "grid_fee": grid_fee_eur,
            "vat": vat_fraction,
            "fix_price_buy": fix_price_buy_eur,
            "fix_price_sell": fix_price_sell_eur,
            "battery_capacity": battery_capacity or 0.0,
            "battery_max_charge": battery_max_charge or 0.0,
            "battery_max_discharge": battery_max_discharge or 0.0,
            "battery_soc_min": battery_soc_min or 0.0,
            "battery_soc_final": battery_soc_final or 0.0,
            "battery_eta_charge": battery_eta_charge or 0.0,
            "battery_eta_discharge": battery_eta_discharge or 0.0,
            "control_algorithm": control_algorithm or "model-predictive-control",
            "allow_feed_in": "yes" not in (allow_feed_in_val or []),
            "objective": opt_objective or "profit",
        }

        # Netload aus Server-Cache statt aus geteilter bess-Instanz
        df_energy = _netload_cache.get(session_id or "") if session_id else None
        if df_energy is None:
            df_energy = bess.get_netload_profile()

        thread = threading.Thread(
            target=_run_year_sim_job,
            args=(start_ts, end_ts, params, df_energy, session_id or ""),
            daemon=True,
        )
        thread.start()

        return (
            "0",
            f"Starte Simulation fuer {len(days)} Tage...",
            "",
            no_update,
        )

    # ── Progress-Callback: WebSocket-Push → Store → UI ──────────────
    @app.callback(
        Output("year-progress", "value"),
        Output("year-progress-text", "children"),
        Output("year-summary", "children"),
        Output("year-graph", "figure"),
        Input("ws-sim-progress", "data"),
        prevent_initial_call=True,
    )
    def on_sim_progress(data):
        if not data:
            raise PreventUpdate

        status = data.get("status", "running")
        completed_days = int(data.get("completed_days", 0))
        total_days = int(data.get("total_days", 0))
        progress = int(data.get("progress", 0))
        rows = list(data.get("rows", []))
        last_day = data.get("last_day")

        if status == "error":
            return (
                str(progress),
                f"Fehler bei Tag {completed_days}/{total_days}.",
                f"Fehler: {data.get('error', 'Unbekannt')}",
                no_update,
            )

        if status == "running":
            progress_text = (
                f"Tag {completed_days}/{total_days} gerechnet ({last_day})"
                if completed_days > 0 and last_day
                else f"Simulation laeuft... 0/{total_days}"
            )
            return (
                str(progress),
                progress_text,
                no_update,
                no_update,
            )

        # status == "done"
        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df["date"] = pd.to_datetime(result_df["date"])
            result_df = result_df.set_index("date").sort_index()

        year_profit = float(result_df["profit_eur"].sum()) if not result_df.empty else 0.0
        import_kwh = float(result_df["grid_import_kwh"].sum()) if not result_df.empty else 0.0
        export_kwh = float(result_df["grid_export_kwh"].sum()) if not result_df.empty else 0.0
        throughput_kwh = float(result_df["battery_throughput_kwh"].sum()) if (not result_df.empty and "battery_throughput_kwh" in result_df.columns) else 0.0
        battery_capacity = float(data.get("battery_capacity", 0.0))
        cycles = throughput_kwh / battery_capacity if battery_capacity > 0 else 0.0
        summary = (
            f"Jahresgewinn: {year_profit:.2f} EUR | "
            f"Netzbezug: {import_kwh:.1f} kWh | "
            f"Einspeisung: {export_kwh:.1f} kWh | "
            f"Batteriezyklen: {cycles:.0f}"
        )

        return (
            "100",
            f"Abgeschlossen. {completed_days} Tage berechnet.",
            summary,
            build_year_figure(result_df),
        )

    # ── Client-side: SocketIO mit Session-Room verbinden ───────────
    app.clientside_callback(
        """
        function(sessionId) {
            if (!sessionId || window._bessSocket) return window.dash_clientside.no_update;
            var socket = io();
            window._bessSocket = socket;
            socket.emit("join", {session_id: sessionId});
            socket.on("sim_progress", function(data) {
                dash_clientside.set_props("ws-sim-progress", {data: data});
            });
            return true;
        }
        """,
        Output("socket-init", "data"),
        Input("session-id", "data"),
    )

    # Für Gunicorn: kein direkter Start, sondern Rückgabe der App
    return app


def create_application():
    """WSGI application factory. Used by Gunicorn."""
    bess = Bess()
    app = run_dashboard(bess)
    return app.server


if __name__ == "__main__":
    application = create_application()
    application.run(debug=True, port=8500)
