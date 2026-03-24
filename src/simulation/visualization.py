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
            "EPEX Day-Ahead Preis",
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


def parse_vkw_csv(contents: str) -> pd.DataFrame:
    """Parst VKW-Smartmeter-CSV (4 Header-Zeilen, Semikolon,
    Spalten: Beginn;Ende;Messwert)."""
    _, content_string = contents.split(",", maxsplit=1)
    decoded = base64.b64decode(content_string).decode("utf-8")
    df = pd.read_csv(
        StringIO(decoded),
        sep=";",
        skiprows=4,
        decimal=",",
    )
    df.columns = df.columns.str.strip()
    df["ts"] = pd.to_datetime(
        df["Beginn der Messung"],
        format="%d.%m.%Y %H:%M:%S",
    )
    df["ts"] = df["ts"].dt.tz_localize("Europe/Vienna")
    df = df.set_index("ts")
    df["value_kw"] = pd.to_numeric(
        df["Messwert"], errors="coerce").fillna(0.0)
    return df[["value_kw"]]


def run_dashboard(
    bess: Bess,
    use_dynamic_prices: bool = True,
    port: int = 8051,
) -> None:

    print(f"Dash app running on http://127.0.0.1:{port}")

    app = Dash(__name__)

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
                    "Model Predictive Control Simulation",
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
                                html.Div("Daten", style=CARD_TITLE),
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
                                # Upload: Residuallast (1 CSV)
                                html.Div(
                                    id="upload-residual",
                                    children=[
                                        dcc.Upload(
                                            id="residual-profile-upload",
                                            children=html.Button(
                                                "Residuallast CSV",
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
                                                "Last CSV",
                                                style={**BTN,
                                                       "width": "100%"}),
                                            multiple=False,
                                        ),
                                        html.Div(style={"height": "6px"}),
                                        dcc.Upload(
                                            id="gen-profile-upload",
                                            children=html.Button(
                                                "Erzeugung CSV",
                                                style={**BTN,
                                                       "width": "100%"}),
                                            multiple=False,
                                        ),
                                    ],
                                ),
                            ]),

                            # Card: Optimierungsziel
                            html.Div(style=CARD, children=[
                                html.Div("Optimierungsziel",
                                         style=CARD_TITLE),
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

                            # Card: Einstellungen
                            html.Div(style=CARD, children=[
                                html.Div("Einstellungen",
                                         style=CARD_TITLE),
                                dcc.Checklist(
                                    id="allow-feed-in",
                                    options=[{
                                        "label": " PV-Einspeisung erlaubt",
                                        "value": "yes",
                                    }],
                                    value=["yes"],
                                    labelStyle={
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                                html.Hr(style={
                                    "border": "none",
                                    "borderTop": f"1px solid {COLOR['border']}",
                                    "margin": "12px 0"}),
                                html.Div("Preismodell",
                                         style={**CARD_TITLE,
                                                "marginTop": "4px"}),
                                dcc.RadioItems(
                                    id="price-source",
                                    options=[
                                        {"label": " EPEX Day-Ahead",
                                         "value": "epex"},
                                        {"label": " Time-of-Use",
                                         "value": "time-of-use",
                                         "disabled": True},
                                        {"label": " Fix",
                                         "value": "fix",
                                         "disabled": True},
                                    ],
                                    value="epex",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "5px",
                                        "fontSize": "14px",
                                        "cursor": "pointer"},
                                ),
                            ]),

                            # Card: Steuerung
                            html.Div(style=CARD, children=[
                                html.Div("Steuerung", style=CARD_TITLE),
                                dcc.RadioItems(
                                    id="control-algorithm",
                                    options=[
                                        {"label": " PV-\u00dcberschussladen",
                                         "value": "pv-ueberschussladen",
                                         "disabled": True},
                                        {"label": " Laden ab 2 kW",
                                         "value": "time-of-use",
                                         "disabled": True},
                                        {"label": " MPC",
                                         "value": "model-predictive-control"},
                                    ],
                                    value="model-predictive-control",
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
                                                id="result-graph"),
                                        ),
                                    ],
                                ),

                                # ── Tab 2: Jahressimulation ──────────
                                dcc.Tab(
                                    label="Jahressimulation",
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
        Output("act-day-picker", "date"),
        Input("prev-day", "n_clicks"),
        Input("next-day", "n_clicks"),
        State("act-day-picker", "date"),
    )
    def shift_day(prev_clicks, next_clicks, current_date):
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

        date = max(pd.Timestamp("2025-01-01"), date)
        date = min(pd.Timestamp("2025-12-31"), date)

        return date.date()

    @app.callback(
        Output("result-graph", "figure"),
        Input("act-day-picker", "date"),
        Input("residual-profile-upload", "contents"),
        Input("load-profile-upload", "contents"),
        Input("gen-profile-upload", "contents"),
        State("input-mode", "value"),
    )
    def update_graph(act_day, residual_contents, load_contents,
                     gen_contents, input_mode):

        if input_mode == "load_gen":
            if load_contents and gen_contents:
                df_load = parse_vkw_csv(load_contents)
                df_gen = parse_vkw_csv(gen_contents)
                net = df_load["value_kw"] - df_gen["value_kw"]
                bess.df_energy = pd.DataFrame(
                    {"net_load_kw": net})
        elif residual_contents:
            df_res = parse_vkw_csv(residual_contents)
            bess.df_energy = pd.DataFrame(
                {"net_load_kw": df_res["value_kw"]})

        act_day = pd.Timestamp(act_day, tz="Europe/Vienna")
        bess.run(act_day=act_day, use_dynamic_prices=use_dynamic_prices,
                 verbose=False)
        return build_figure(bess)

    app.run(debug=True, port=port)


if __name__ == "__main__":
    bess = Bess()
    run_dashboard(bess)
