import pickle

import pandas as pd


class ForecastingModel:
    def __init__(self, filename="gam_model.pkl"):

        with open(filename, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, x):

        return self.model.predict(x)

    def train_model(self):

        # Read in the historic load profile
        df_load = pd.read_csv("../interfaces/energy_data.csv", index_col=0)
        df_load.index = pd.to_datetime(df_load.index, utc=True)
        df_load.index = df_load.index.tz_convert("Europe/Vienna")
        net_load_profile = df_load["Consumption"] - df_load["Production"]
        net_load_profile = net_load_profile / 1000.0  # Umrechnung in kW
        net_load_profile = net_load_profile["2025-01-01":"2025-12-31"]
