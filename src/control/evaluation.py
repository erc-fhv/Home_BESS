from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

class MpcEvaluationDashboard:
	def __init__(self, output_dir: str | Path = "", file_pattern: str = "mpc_results_*.parquet") -> None:

		if output_dir == "":
			output_dir = Path(__file__).parent / "output"
			if not output_dir.exists():
				raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")

		self.output_dir = Path(output_dir)
		self.file_pattern = file_pattern
		self._df_cache: dict[str, pd.DataFrame] = {}

	def _get_file_options_and_default(self) -> tuple[list[dict[str, str]], str]:
		parquet_files = sorted(self.output_dir.glob(self.file_pattern))
		if not parquet_files:
			raise FileNotFoundError(
				f"No parquet files found in {self.output_dir}/ matching pattern '{self.file_pattern}'"
			)

		default_file = str(max(parquet_files, key=lambda path: path.stat().st_mtime).resolve())
		file_options = [{"label": path.name, "value": str(path.resolve())} for path in parquet_files]
		return file_options, default_file

	def _load_df(self, selected_file: str) -> pd.DataFrame:
		if selected_file in self._df_cache:
			return self._df_cache[selected_file]

		df = pd.read_parquet(selected_file).copy()
		df["mpc_time"] = pd.to_datetime(df["mpc_time"])
		df["timestamp"] = pd.to_datetime(df["timestamp"])
		self._df_cache[selected_file] = df
		return df

	def _get_mpc_time_options(self, selected_file: str) -> tuple[list[dict[str, str]], str | None]:
		df = self._load_df(selected_file)
		mpc_times = sorted(pd.to_datetime(df["mpc_time"]).unique())
		options = [
			{
				"label": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
				"value": pd.Timestamp(ts).isoformat(),
			}
			for ts in mpc_times
		]
		default_value = options[-1]["value"] if options else None
		return options, default_value

	def _build_plot_for_selection(self, selected_file: str, selected_mpc_time_iso: str | None) -> go.Figure:
		if selected_mpc_time_iso is None:
			return go.Figure()

		df = self._load_df(selected_file)
		selected_mpc_time = pd.Timestamp(selected_mpc_time_iso)
		mpc_result = df[df["mpc_time"] == selected_mpc_time].copy()
		mpc_result = mpc_result.sort_values("timestamp").set_index("timestamp")
		return self._build_figure(mpc_result, selected_mpc_time)

	def run(self, host: str = "127.0.0.1", debug: bool = False) -> None:
		file_options, default_file = self._get_file_options_and_default()
		initial_mpc_options, initial_mpc_value = self._get_mpc_time_options(default_file)

		app = Dash(__name__)
		app.layout = html.Div(
			[
				html.H3("MPC Control Results"),
				html.Label("File:"),
				dcc.Dropdown(id="file-dropdown", options=file_options, value=default_file, clearable=False),
				html.Br(),
				html.Label("MPC Run:"),
				dcc.Dropdown(
					id="mpc-time-dropdown",
					options=initial_mpc_options,
					value=initial_mpc_value,
					clearable=False,
				),
				html.Br(),
				dcc.Graph(id="mpc-graph", figure=self._build_plot_for_selection(default_file, initial_mpc_value)),
			],
			style={"maxWidth": "1100px", "margin": "20px auto", "padding": "0 16px"},
		)

		@app.callback(
			Output("mpc-time-dropdown", "options"),
			Output("mpc-time-dropdown", "value"),
			Input("file-dropdown", "value"),
		)
		def _update_mpc_time_dropdown(selected_file: str):
			return self._get_mpc_time_options(selected_file)

		@app.callback(
			Output("mpc-graph", "figure"),
			Input("file-dropdown", "value"),
			Input("mpc-time-dropdown", "value"),
		)
		def _update_graph(selected_file: str, selected_mpc_time_iso: str | None):
			return self._build_plot_for_selection(selected_file, selected_mpc_time_iso)

		app.run(host=host, debug=debug)

	def show(self) -> None:
		self.run()

	def _build_figure(self, mpc_result: pd.DataFrame, selected_mpc_time: pd.Timestamp):
		fig = make_subplots(
			rows=4,
			cols=1,
			shared_xaxes=True,
			vertical_spacing=0.03,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["price_buy_eur_kwh"],
				name="Buy price",
				mode="lines",
				line=dict(color="#2ca02c", width=2, shape="hv"),
				showlegend=False,
			),
			row=1,
			col=1,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["price_sell_eur_kwh"],
				name="Sell price",
				mode="lines",
				line=dict(color="#2c92a0", width=2, shape="hv"),
				showlegend=False,
			),
			row=1,
			col=1,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["netload_forecast_kw"],
				name="Netload Forecast",
				mode="lines",
				line=dict(color="#1f77b4", width=2, shape="hv"),
				showlegend=False,
			),
			row=2,
			col=1,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["soc_percent"],
				mode="lines",
				name="Battery SOC",
				line=dict(color="#9467bd", width=2, shape="hv"),
				showlegend=False,
			),
			row=3,
			col=1,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["set_netload_kw"],
				name="Grid Power Setpoint",
				showlegend=False,
				mode="lines",
				line=dict(color="#ff7f0e", width=2, shape="hv"),
			),
			row=4,
			col=1,
		)

		fig.update_yaxes(title_text="Energy<br>Price [€/kWh]", title_standoff=14, automargin=True, row=1, col=1)
		fig.update_yaxes(title_text="Netload<br>Forecast [kW]", title_standoff=14, automargin=True, row=2, col=1)
		fig.update_yaxes(title_text="Battery<br>SOC [%]", title_standoff=14, automargin=True, row=3, col=1)
		fig.update_yaxes(title_text="Grid<br>Power [kW]", title_standoff=14, automargin=True, row=4, col=1)
		fig.update_xaxes(title_text="Uhrzeit", tickformat="%H:%M", row=4, col=1)

		fig.update_layout(
			height=780,
			template="plotly_white",
			hovermode="x unified",
			margin=dict(l=120, r=30, t=60, b=50),
			showlegend=True,
		)
		return fig

if __name__ == "__main__":
	dashboard = MpcEvaluationDashboard()
	dashboard.show()
