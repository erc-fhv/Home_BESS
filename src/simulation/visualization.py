import base64
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
from dash import Dash, dcc, html, Input, Output, callback_context, State

from simulation.bess_simulation import Bess

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


def build_figure(bess: Bess) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Strom Preis (brutto)",
            "Residuallast (= Last - PV)",
            "Batterie State of Charge",
            "Netz Einspeisung / Bezug",
            "Batterie Laden / Entladen",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=bess.act_prices_epex.index,
            y=bess.act_prices_epex.values,
            line_shape="hv",
            name="Epex Preis",
            legendgroup="g1", legend="legend",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bess.act_prices_epex.index,
            y=bess.price_sell.values,
            line_shape="hv",
            name="Einspeisepreis",
            legendgroup="g1", legend="legend",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bess.act_prices_epex.index,
            y=bess.price_buy.values,
            line_shape="hv",
            name="Bezugspreise",
            legendgroup="g1", legend="legend",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bess.net_load_kw.index,
            y=bess.net_load_kw.values,
            line_shape="hv",
            name="Residuallast",
            legendgroup="g2", legend="legend2",
        ),
        row=2, col=1,
    )

    soc_percent = 100 * bess.lp_result["soc"] / bess.capacity_kwh
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
            x=bess.lp_result["p_sell"].index,
            y=bess.lp_result["p_sell"].values,
            line_shape="hv",
            name="Einspeisung",
            legendgroup="g4", legend="legend4",
        ),
        row=4, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bess.lp_result["p_buy"].index,
            y=bess.lp_result["p_buy"].values,
            line_shape="hv",
            name="Bezug",
            legendgroup="g4", legend="legend4",
        ),
        row=4, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bess.lp_result["p_ch"].index,
            y=bess.lp_result["p_ch"].values,
            line_shape="hv",
            name="Laden",
            legendgroup="g5", legend="legend5",
        ),
        row=5, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bess.lp_result["p_dis"].index,
            y=bess.lp_result["p_dis"].values,
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

    try:
        objective_value = pulp.value(bess.pulp_model.objective)
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


def parse_csv(contents: str) -> pd.DataFrame:
    """Parst hochgeladene CSV-Dateien.

    Erkennt automatisch:
    - VKW-Smartmeter-Format (4 Header-Zeilen, Semikolon, Latin-1)
    - Einfaches CSV (1 Header-Zeile, erste Spalte=Datetime, zweite=kW)
    """
    _, content_string = contents.split(",", maxsplit=1)
    raw = base64.b64decode(content_string)

    # Versuche Latin-1 (VKW) und UTF-8
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw.decode("latin-1")

    lines = decoded.strip().splitlines()
    first_line = lines[0].strip()

    # VKW-Format: erste Zeile beginnt nicht mit typischen CSV-Headern
    # und enthält Semikolons in den Datenzeilen
    is_vkw = ";" in lines[min(4, len(lines) - 1)] and "Messwert" in decoded

    if is_vkw:
        df = pd.read_csv(
            StringIO(decoded), sep=";", skiprows=4, decimal=",",
        )
        df.columns = df.columns.str.strip()
        df["ts"] = pd.to_datetime(
            df["Beginn der Messung"], format="%d.%m.%Y %H:%M:%S",
        )
        df["ts"] = df["ts"].dt.tz_localize("Europe/Vienna")
        df = df.set_index("ts")
        df["value_kw"] = pd.to_numeric(
            df["Messwert"], errors="coerce").fillna(0.0)
    else:
        # Einfaches CSV: Spalte 1 = Datetime, Spalte 2 = Wert in kW
        df = pd.read_csv(StringIO(decoded))
        df.columns = df.columns.str.strip()
        ts_col = df.columns[0]
        val_col = df.columns[1]
        df["ts"] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(
            "Europe/Vienna")
        df = df.set_index("ts")
        df["value_kw"] = pd.to_numeric(
            df[val_col], errors="coerce").fillna(0.0)

    return df[["value_kw"]]


def run_dashboard(
    bess: Bess,
    use_dynamic_prices: bool = True,
    port: int = 8051,
) -> None:

    print(f"Dash app running on http://127.0.0.1:{port}")

    app = Dash(__name__, title="FHV FZE - Lastmanagement Simulation")

    # Verfügbare Tage aus dem aktuellen Datensatz
    _first_date = bess.df_energy.index.min().normalize()
    _last_date = bess.df_energy.index.max().normalize()

    # ── Layout ───────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={"backgroundColor": COLOR["bg"], "minHeight": "100vh",
                "fontFamily": "Inter, system-ui, -apple-system, sans-serif"},
        children=[
            # ── Header ───────────────────────────────────────────────────
            html.Div(
                style={"backgroundColor": COLOR["header"],
                       "padding": "18px 40px",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.12)"},
                children=html.H1(
                    "FHV FZE - Lastmanagement Simulation",
                    style={"color": "#fff", "margin": "0",
                           "fontSize": "22px", "fontWeight": "700",
                           "letterSpacing": "-0.3px"},
                ),
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
                                        {"label": " Last + Erzeugung",
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
                                            ("Profil (ohne Batterieeinfluss) als CSV hochladen. ",
                                             "Erste Spalte Datum/Uhrzeit, zweite Spalte kW-Werte."),
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
                                        html.Div(style={"height": "6px"}),
                                        dcc.Upload(
                                            id="gen-profile-upload",
                                            children=html.Button(
                                                "Upload PV",
                                                style={**BTN,
                                                       "width": "100%"}),
                                            multiple=False,
                                        ),
                                    ],
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
                                            value=12.72, step=0.01,
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
                                            value=9.0, step=0.1,
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
                                    value=6.0, step=0.1,
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
                                    value=20.0, step=0.1,
                                    style={"width": "100%",
                                           "padding": "6px 10px",
                                           "borderRadius": "6px",
                                           "border": f"1px solid {COLOR['border']}",
                                           "fontSize": "13px"},
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
                                        {"label": " Model Predictive Control",
                                         "value": "model-predictive-control"},
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
                                        "label": " Batterie-Einspeisung verbieten",
                                        "value": "yes",
                                    }],
                                    value=["no"],
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

                                # ── Tab 2: Jahressimulation ──────────
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
                                                html.H3(
                                                    "Jahressimulation",
                                                    style={
                                                        "marginTop": "0",
                                                        "color": COLOR["text"],
                                                    }),
                                                html.P(
                                                    "Hier wird die Jahresoptimierung "
                                                    "dargestellt: kumulierter Gewinn, "
                                                    "Autarkiegrad, Netzbezug und "
                                                    "-einspeisung pro Monat.",
                                                    style={
                                                        "color": COLOR["text_light"],
                                                        "marginBottom": "16px",
                                                    }),
                                                dcc.Graph(
                                                    id="year-graph",
                                                    figure=go.Figure().update_layout(
                                                        height=600,
                                                        template="plotly_white",
                                                        font=dict(
                                                            family="Inter, sans-serif",
                                                            size=12),
                                                        paper_bgcolor="#f8fafc",
                                                        annotations=[dict(
                                                            text="Noch nicht implementiert",
                                                            xref="paper", yref="paper",
                                                            x=0.5, y=0.5,
                                                            showarrow=False,
                                                            font=dict(
                                                                size=20,
                                                                color=COLOR["text_light"]),
                                                        )],
                                                    ),
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
        Input("control-algorithm", "value"),
    )
    def toggle_mpc_options(algorithm):
        disabled = algorithm == "pv-ueberschussladen"
        disabled = True # molu
        feed_in_opts = [{
            "label": " Batterie-Einspeisung verbieten",
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
        return feed_in_opts, obj_opts

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
        State("input-mode", "value"),
        prevent_initial_call=True,
    )
    def update_graph(act_day, residual_contents, load_contents,
                     gen_contents, price_source, epex_offset_buy,
                     epex_offset_sell, grid_fee, vat,
                     fix_price_buy, fix_price_sell,
                     input_mode):

        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        # Nur bei Upload neu parsen, nicht bei Datumswechsel
        if triggered in ("residual-profile-upload", "load-profile-upload", "gen-profile-upload"):
            if input_mode == "load_gen":
                if load_contents and gen_contents:
                    df_load = parse_csv(load_contents)
                    df_gen = parse_csv(gen_contents)
                    net = df_load["value_kw"] - df_gen["value_kw"]
                    bess.df_energy = pd.DataFrame(
                        {"net_load_kw": net})
            elif residual_contents:
                df_res = parse_csv(residual_contents)
                bess.df_energy = pd.DataFrame(
                    {"net_load_kw": df_res["value_kw"]})

        min_date = bess.df_energy.index.min().normalize()
        max_date = bess.df_energy.index.max().normalize()

        act_day = pd.Timestamp(act_day, tz="Europe/Vienna")
        # Bei neuem Upload auf ersten verfuegbaren Tag springen
        if triggered in ("residual-profile-upload", "load-profile-upload", "gen-profile-upload"):
            act_day = min_date if min_date.tzinfo else min_date.tz_localize("Europe/Vienna")

        bess.run(act_day=act_day,
                 use_dynamic_prices=(price_source == "epex"),
                 epex_offset_buy=(epex_offset_buy or 0) / 100.0,
                 epex_offset_sell=(epex_offset_sell or 0) / 100.0,
                 grid_fee=(grid_fee or 0) / 100.0,
                 vat=(vat or 0) / 100.0,
                 fix_price_buy=(fix_price_buy or 0) / 100.0,
                 fix_price_sell=(fix_price_sell or 0) / 100.0,
                 verbose=False)
        return build_figure(bess), act_day.date(), min_date.date(), max_date.date()

    app.run(debug=True, port=port)


if __name__ == "__main__":
    bess = Bess()
    run_dashboard(bess)
