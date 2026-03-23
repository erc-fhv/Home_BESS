from pathlib import Path
import base64
from io import StringIO
import pulp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback_context, State

from interfaces.get_day_ahead_prices import DayAheadPrice

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


class Bess:
    def __init__(
        self,
        capacity_kwh: float = 5.12*6,
        max_charge_kw: float = 4.0*3,
        max_discharge_kw: float = 4.0*3,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        eta_charge: float = np.sqrt(0.95) * 0.96,       # 95% BESS round-trip efficiency
        eta_discharge: float = np.sqrt(0.95) * 0.96,    # times 96% inverter power factor
        ) -> None:

        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.soc_min_kwh = soc_min * capacity_kwh
        self.soc_max_kwh = soc_max * capacity_kwh
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge

        self.lp_result = {}
        self.price_sell = pd.Series()
        self.price_buy = pd.Series()

        # Epex Preise einlesen
        file_path = Path(__file__).parent / "data" / "epex_prices_2025.csv"
        if not file_path.exists():
            DayAheadPrice.get_epex_prices(
                country_code="AT",
                start_date=pd.Timestamp("2025-01-01", tz="Europe/Vienna"),
                end_date=pd.Timestamp("2025-12-24", tz="Europe/Vienna"),
                store_to_file=file_path,
                )
        df_prices_epex = pd.read_csv(file_path, index_col=0)
        df_prices_epex.index = pd.to_datetime(df_prices_epex.index, utc=True)
        df_prices_epex.index = df_prices_epex.index.tz_convert("Europe/Vienna")
        self.prices_epex = df_prices_epex["day_ahead_price_eur_kWh"]
        self.prices_epex = self.prices_epex.resample('15min').ffill()

        # Energieverbrauchs- und Produktionsdaten einlesen
        # todo: ebenfalls automatisch einlesen, falls nicht vorhanden.
        file_path = Path(__file__).parent / "data" / "example_household_without_battery.csv"
        self.df_energy = pd.read_csv(file_path, index_col=0)
        self.df_energy.index = pd.to_datetime(self.df_energy.index, utc=True)
        self.df_energy.index = self.df_energy.index.tz_convert("Europe/Vienna")

    def optimize_day(
        self,
        price_sell: pd.Series,    # ct/kWh
        price_buy: pd.Series,     # ct/kWh
        net_load: pd.Series,      # kW
        soc0: float,              # kWh
        verbose: bool = True,
        ) -> tuple:

        # Konsistenzcheck
        assert price_sell.index.equals(price_buy.index)
        assert price_buy.index.equals(net_load.index)

        # Definiere Zeitindex und Parameter
        time_points = price_sell.index
        T = range(len(time_points))
        time_periods = time_points[:-1]
        P = range(len(time_periods))
        delta_t = (time_points[1] - time_points[0]).total_seconds() / 3600.0  # in hours

        # Definition des Gewinnmaximierungsproblems
        model = pulp.LpProblem("DayAheadOpt", pulp.LpMaximize)

        # Entscheidungsvariablen
        p_ch   = pulp.LpVariable.dicts("p_ch", P, 0, self.max_charge_kw)
        p_dis  = pulp.LpVariable.dicts("p_dis", P, 0, self.max_discharge_kw)
        soc    = pulp.LpVariable.dicts("soc", T, self.soc_min_kwh, self.soc_max_kwh)

        p_sell = pulp.LpVariable.dicts("p_sell", P, 0)
        p_buy  = pulp.LpVariable.dicts("p_buy", P, 0)

        y = pulp.LpVariable.dicts("y", P, 0, 1, cat="Binary")   # Lade-/Entlade-Exklusivität

        # Anfangs- und End-SOC festsetzen
        model += soc[0] == soc0
        model += soc[T[-1]] == soc0

        # Füge SOC Nebenbedingungen hinzu
        for t in range(1, len(T)):
            model += soc[t] == soc[t-1] + \
                delta_t * (self.eta_charge * p_ch[t-1] - p_dis[t-1] / self.eta_discharge)

        # Lade-/Entlade-Exklusivität
        for p in P:
            model += p_ch[p]  <= self.max_charge_kw * y[p]
            model += p_dis[p] <= self.max_discharge_kw * (1 - y[p])

        # Leistungsbilanz
        for p in P:
            model += (
                p_buy[p] + p_dis[p] - net_load.iloc[p] - p_sell[p] - p_ch[p]
                ==
                0.0
            )

        # Zielfunktion: Erlös – Kosten
        model += pulp.lpSum(
            (price_sell.iloc[p] * p_sell[p]
            - price_buy.iloc[p] * p_buy[p]) * delta_t
            for p in P
        )

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if verbose:
            print("Pulp-Resultate:")
            print(f"- Status numerisch: {model.status}")
            print(f"- Status verbal: {pulp.LpStatus[model.status]}")
            print(f"- optimaler Wert: {pulp.value(model.objective)}")

        ret_variables = {
            "soc":    pd.Series([soc[t].value() for t in T], index=time_points),
            "p_ch":   pd.Series([p_ch[p].value() for p in P], index=time_periods),
            "p_dis":  pd.Series([p_dis[p].value() for p in P], index=time_periods),
            "p_sell": pd.Series([p_sell[p].value() for p in P], index=time_periods),
            "p_buy":  pd.Series([p_buy[p].value() for p in P], index=time_periods),
        }

        return ret_variables, model

    def run(
        self,
        act_day: pd.Timestamp,
        use_dynamic_prices: bool = True,
        verbose: bool = False,
        ) -> None:

        act_range = pd.date_range(
            start=act_day,
            end=act_day + pd.Timedelta(days=1),
            freq='15min',
            tz="Europe/Vienna",
            )
        self.act_prices_epex = self.prices_epex.loc[act_range]

        if use_dynamic_prices:
            # VKW dynmaische Preise in EUR/kWh
            self.price_sell = self.act_prices_epex - 0.006
            self.price_buy  = self.act_prices_epex + 0.0144
        else:
            # fixe Preise in ct/kWh
            self.price_sell = pd.Series(0.09, index=self.act_prices_epex.index)
            self.price_buy  = pd.Series(0.1272, index=self.act_prices_epex.index)

        self.net_load_kw = self.df_energy.loc[act_range]["net_load_kw"]

        self.lp_result, self.pulp_model = self.optimize_day(
            price_sell=self.price_sell,
            price_buy=self.price_buy,
            net_load=self.net_load_kw,
            soc0=0.5 * self.capacity_kwh,
            verbose=verbose,
        )

    def build_figure(self):
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
                x=self.act_prices_epex.index,
                y=self.act_prices_epex.values,
                line_shape="hv",
                name="Epex Preis",
                legendgroup="g1", legend="legend",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.act_prices_epex.index,
                y=self.price_sell.values,
                line_shape="hv",
                name="Einspeisepreis",
                legendgroup="g1", legend="legend",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.act_prices_epex.index,
                y=self.price_buy.values,
                line_shape="hv",
                name="Bezugspreise",
                legendgroup="g1", legend="legend",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.net_load_kw.index,
                y=self.net_load_kw.values,
                line_shape="hv",
                name="Residuallast",
                legendgroup="g2", legend="legend2",
            ),
            row=2, col=1,
        )

        soc_percent = 100 * self.lp_result["soc"] / self.capacity_kwh
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
                x=self.lp_result["p_sell"].index,
                y=self.lp_result["p_sell"].values,
                line_shape="hv",
                name="Einspeisung",
                legendgroup="g4", legend="legend4",
            ),
            row=4, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_buy"].index,
                y=self.lp_result["p_buy"].values,
                line_shape="hv",
                name="Bezug",
                legendgroup="g4", legend="legend4",
            ),
            row=4, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_ch"].index,
                y=self.lp_result["p_ch"].values,
                line_shape="hv",
                name="Laden",
                legendgroup="g5", legend="legend5",
            ),
            row=5, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_dis"].index,
                y=self.lp_result["p_dis"].values,
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
            # Legenden pro Subplot
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

        # --- Y-Achsen hinzufügen ---
        fig.update_yaxes(
            title_text="Preis [EUR/kWh]",
            row=1, col=1,
        )

        fig.update_yaxes(
            title_text="Leistung [kW]",
            row=2, col=1,
        )

        fig.update_yaxes(
            title_text="SOC [%]",
            range=[0, 100],
            row=3, col=1,
        )

        fig.update_yaxes(
            title_text="Leistung [kW]",
            row=4, col=1,
        )

        fig.update_yaxes(
            title_text="Leistung [kW]",
            row=5, col=1,
        )

        # Anzeige des aktuellen Objective-Werts (falls verfügbar) oben rechts
        try:
            objective_value = pulp.value(self.pulp_model.objective)
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
            # Falls das Modell noch nicht gesetzt ist, nichts tun
            pass

        return fig

    def run_dashboard(
        self,
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
                                # Card: Eingabemodus
                                html.Div(style=CARD, children=[
                                    html.Div("Eingabemodus", style=CARD_TITLE),
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
                                    dcc.Upload(
                                        id="residual-profile-upload",
                                        children=html.Button(
                                            "CSV hochladen",
                                            style={**BTN, "width": "100%"}),
                                        multiple=False,
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

            # Begrenzung auf 2025
            date = max(pd.Timestamp("2025-01-01"), date)
            date = min(pd.Timestamp("2025-12-31"), date)

            return date.date()

        @app.callback(
            Output("result-graph", "figure"),
            Input("act-day-picker", "date"),
            Input("residual-profile-upload", "contents"),
        )
        def update_graph(act_day, residual_profile_upload_contents):
            if residual_profile_upload_contents:
                _, content_string = residual_profile_upload_contents.split(",", maxsplit=1)
                decoded = base64.b64decode(content_string).decode("utf-8")
                self.df_energy = pd.read_csv(StringIO(decoded), index_col=0)
                energy_index = pd.DatetimeIndex(pd.to_datetime(self.df_energy.index, utc=True))
                self.df_energy.index = energy_index.tz_convert("Europe/Vienna")
                self.df_energy = self.df_energy / 1000.0

            act_day = pd.Timestamp(act_day, tz="Europe/Vienna")
            self.run(act_day=act_day, use_dynamic_prices=use_dynamic_prices, verbose=False)
            return self.build_figure()

        app.run(debug=True, port=port)

if __name__ == "__main__":
    bess = Bess()
    bess.run_dashboard()
