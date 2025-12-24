import pulp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback_context, State

class Bess:
    def __init__(
        self,
        capacity_kwh: float = 5.12*4,
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
        df_prices_epex = pd.read_csv("../data/day_ahead_prices.csv", index_col=0)
        df_prices_epex.index = pd.to_datetime(df_prices_epex.index, utc=True)
        df_prices_epex.index = df_prices_epex.index.tz_convert("Europe/Vienna")
        self.prices_epex = df_prices_epex["day_ahead_price_EUR_MWh"]
        self.prices_epex = self.prices_epex / 1000  # Umrechnung in EUR/kWh
        self.prices_epex = self.prices_epex.resample('15min').ffill()

        # Energieverbrauchs- und Produktionsdaten einlesen
        self.df_energy = pd.read_csv("../data/energy_data.csv", index_col=0)
        self.df_energy.index = pd.to_datetime(self.df_energy.index, utc=True)
        self.df_energy.index = self.df_energy.index.tz_convert("Europe/Vienna")
        self.df_energy = self.df_energy / 1000.0  # Umrechnung in kW

    def optimize_day(
        self,
        price_sell: pd.Series,    # ct/kWh
        price_buy: pd.Series,     # ct/kWh
        pv: pd.Series,            # kW
        load: pd.Series,          # kW
        soc0: float,              # kWh
        verbose: bool = True,
        ) -> tuple:

        # Konsistenzcheck
        assert price_sell.index.equals(price_buy.index)
        assert price_buy.index.equals(pv.index)
        assert pv.index.equals(load.index)

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
                pv.iloc[p] + p_buy[p] + p_dis[p]
                ==
                load.iloc[p] + p_sell[p] + p_ch[p]
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

        self.pv_forecast = self.df_energy.loc[act_range]["Production"]
        self.load_forecast = self.df_energy.loc[act_range]["Consumption"]
        self.net_load = self.load_forecast - self.pv_forecast

        self.lp_result, self.pulp_model = self.optimize_day(
            price_sell=self.price_sell,
            price_buy=self.price_buy,
            pv=self.pv_forecast,
            load=self.load_forecast,
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
                "Residuallastprofil",
                "Batterie State of Charge",
                "Netz Einspeisung / Bezug",
                "Batterieladung",
            ],
        )

        fig.add_trace(
            go.Scatter(
                x=self.act_prices_epex.index,
                y=self.act_prices_epex.values,
                line_shape="hv",
                name="Epex Preis",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.act_prices_epex.index,
                y=self.price_sell.values,
                line_shape="hv",
                name="Einspeisepreis",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.act_prices_epex.index,
                y=self.price_buy.values,
                line_shape="hv",
                name="Bezugspreise",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.net_load.index,
                y=self.net_load.values,
                line_shape="hv",
                name="Residuallast",
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
            ),
            row=3, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_sell"].index,
                y=self.lp_result["p_sell"].values,
                line_shape="hv",
                name="Einspeisung",
            ),
            row=4, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_buy"].index,
                y=self.lp_result["p_buy"].values,
                line_shape="hv",
                name="Bezug",
            ),
            row=4, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_ch"].index,
                y=self.lp_result["p_ch"].values,
                line_shape="hv",
                name="Laden",
            ),
            row=5, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.lp_result["p_dis"].index,
                y=self.lp_result["p_dis"].values,
                line_shape="hv",
                name="Entladen",
            ),
            row=5, col=1,
        )

        fig.update_layout(
            height=900,
            hovermode="x unified",
            template="plotly_white",
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
            title_text="Residuallast [kW]",
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
            title_text="Batterie Laden [kW]",
            row=5, col=1,
        )


        return fig

    def run_dashboard(self):

        app = Dash(__name__)

        app.layout = html.Div(
            [
                html.H3("Batterie-Optimierung – Tagesauswahl"),

                html.Div(
                    [
                        html.Button("◀", id="prev-day", n_clicks=0),
                        dcc.DatePickerSingle(
                            id="act-day-picker",
                            min_date_allowed="2025-01-01",
                            max_date_allowed="2025-12-31",
                            date="2025-11-05",
                            display_format="DD.MM.YYYY",
                        ),
                        html.Button("▶", id="next-day", n_clicks=0),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "10px",
                        "marginBottom": "10px",
                    },
                ),

                dcc.Graph(id="result-graph"),
            ],
            style={"width": "1200px", "margin": "auto"},
        )

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
        )
        def update_graph(act_day):
            act_day = pd.Timestamp(act_day, tz="Europe/Vienna")
            self.run(act_day)
            return self.build_figure()

        app.run(debug=True)
