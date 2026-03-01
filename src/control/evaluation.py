from pathlib import Path

import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots


class MpcEvaluationDashboard:
	def __init__(self, output_dir: str | Path = "output", file_pattern: str = "mpc_results_*.parquet") -> None:
		self.output_dir = Path(output_dir)
		self.file_pattern = file_pattern
		self.state: dict[str, pd.DataFrame | None] = {"df": None}

		self.file_dropdown = widgets.Dropdown(
			options=[],
			description="Parquet:",
			layout=widgets.Layout(width="900px"),
		)
		self.mpc_time_dropdown = widgets.Dropdown(
			options=[],
			description="mpc_time:",
			layout=widgets.Layout(width="400px"),
		)
		self.plot_output = widgets.Output()

	def show(self) -> None:
		parquet_files = sorted(self.output_dir.glob(self.file_pattern))
		if not parquet_files:
			raise FileNotFoundError(
				f"No parquet files found in output/ matching pattern '{self.file_pattern}'"
			)

		default_file = max(parquet_files, key=lambda path: path.stat().st_mtime).resolve()
		file_options = [str(path.resolve()) for path in parquet_files]

		self.file_dropdown.options = file_options
		self.file_dropdown.value = str(default_file)

		self.file_dropdown.observe(self._on_file_change, names="value")
		self.mpc_time_dropdown.observe(self._render_plot, names="value")

		self._on_file_change({"name": "value", "new": self.file_dropdown.value})
		display(widgets.VBox([self.file_dropdown, self.mpc_time_dropdown, self.plot_output]))

	def _load_df(self, selected_file: str) -> pd.DataFrame:
		df = pd.read_parquet(selected_file).copy()
		df["mpc_time"] = pd.to_datetime(df["mpc_time"])
		df["timestamp"] = pd.to_datetime(df["timestamp"])
		return df

	def _on_file_change(self, change) -> None:
		if change.get("name") != "value":
			return

		df = self._load_df(change["new"])
		self.state["df"] = df

		mpc_times = sorted(df["mpc_time"].unique())
		self.mpc_time_dropdown.options = mpc_times
		self.mpc_time_dropdown.value = max(mpc_times)

	def _render_plot(self, _=None) -> None:
		df = self.state["df"]
		if df is None or self.mpc_time_dropdown.value is None:
			return

		selected_mpc_time = pd.Timestamp(self.mpc_time_dropdown.value)
		mpc_result = df[df["mpc_time"] == selected_mpc_time].copy()
		mpc_result = mpc_result.sort_values("timestamp").set_index("timestamp")

		with self.plot_output:
			self.plot_output.clear_output(wait=True)
			fig = self._build_figure(mpc_result, selected_mpc_time)
			fig.show()

	def _build_figure(self, mpc_result: pd.DataFrame, selected_mpc_time: pd.Timestamp):
		fig = make_subplots(
			rows=4,
			cols=1,
			shared_xaxes=True,
			vertical_spacing=0.03,
			subplot_titles=(
				"Netload Forecast [kW]",
				"Set Netload [kW]",
				"Buy Price [€/kWh]",
				"SOC [%]",
			),
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["netload_forecast_kw"],
				mode="lines",
				line=dict(color="#1f77b4", width=2, shape="hv"),
			),
			row=1,
			col=1,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["set_netload_kw"],
				mode="lines",
				line=dict(color="#ff7f0e", width=2, shape="hv"),
			),
			row=2,
			col=1,
		)

		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result["price_buy_eur_kwh"],
				mode="lines",
				line=dict(color="#2ca02c", width=2, shape="hv"),
			),
			row=3,
			col=1,
		)

		soc_column = "soc_kwh" if "soc_kwh" in mpc_result.columns else "soc_percent"
		soc_title = "SOC [kWh]" if soc_column == "soc_kwh" else "SOC [%]"
		fig.add_trace(
			go.Scatter(
				x=mpc_result.index,
				y=mpc_result[soc_column],
				mode="lines",
				line=dict(color="#9467bd", width=2, shape="hv"),
			),
			row=4,
			col=1,
		)

		fig.update_yaxes(title_text="Forecast [kW]", row=1, col=1)
		fig.update_yaxes(title_text="Setpoint [kW]", row=2, col=1)
		fig.update_yaxes(title_text="Price [€/kWh]", row=3, col=1)
		fig.update_yaxes(title_text=soc_title, row=4, col=1)
		fig.update_xaxes(title_text="Uhrzeit", tickformat="%H:%M", row=4, col=1)

		fig.update_layout(
			title="MPC Control Results",
			height=780,
			template="plotly_white",
			hovermode="x unified",
			showlegend=False,
		)
		return fig
