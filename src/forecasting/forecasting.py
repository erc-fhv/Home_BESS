import os
from pathlib import Path
import pickle
import pandas as pd
from pygam import s, te, f
import torch
import matplotlib.pyplot as plt

import interfaces.get_weather_data as get_weather_data
from loadforecasting_models import Gam

class ForecastingModel:
    def __init__(self, filename:str=""):

        if filename == "":
            filename = Path(__file__).parent / "gam_model.pkl"

        self.model_filename = filename
        self.lagged_power = 7*24*4    # 7 Tage * 24 Stunden * 4 (15-Minuten-Intervalle)
        self.use_lagged_power = False

    def predict(self, weather_data: pd.DataFrame) -> pd.Series:

        # Load the trained model, if it exists.
        if not os.path.exists(self.model_filename):
            raise FileNotFoundError(
                f"Model file {self.model_filename} not found. Please train the model first.")
        else:
            with open(self.model_filename, "rb") as file:
                model = pickle.load(file)

        # Create features for the prediction
        features_df, _ = self.create_feature_df(weather_data)

        x = features_df.values
        print("molu:", x.shape)
        prediction = model.predict(x)

        return pd.Series(prediction, index=features_df.index)

    def create_feature_df(
        self,
        weather_data: pd.DataFrame,
        net_load_profile: pd.Series = None,
        ) -> tuple[pd.DataFrame, pd.Series]:

        features_df = pd.DataFrame({
            'hour_of_day': weather_data.index.hour,
            'day_of_week': weather_data.index.dayofweek,
            'temperature_2m': weather_data['temperature_2m'],
            'wind_speed_10m': weather_data['wind_speed_10m'],
            'global_tilted_irradiance': weather_data['global_tilted_irradiance'],
        })

        if self.use_lagged_power:
            features_df['lag_7_days'] = net_load_profile.shift(self.lagged_power)

        return features_df, net_load_profile

    def train_model(self):

        # Read in the historic load profile
        data_file = Path(__file__).resolve().parent.parent / "interfaces" / "energy_data.csv"
        df_load = pd.read_csv(data_file, index_col=0)
        df_load.index = pd.to_datetime(df_load.index, utc=True)
        df_load.index = df_load.index.tz_convert('Europe/Vienna')
        net_load_profile = df_load['Consumption'] - df_load['Production']
        net_load_profile = net_load_profile / 1000.0  # Convert from [W] to [kW]
        net_load_profile = net_load_profile['2025-01-01':'2025-12-31']

        # Read in the historic weather data
        weather_data = get_weather_data.WeatherDataRetriever().retrieve_weather_data(
            time_range=net_load_profile.index,
            weather_actuality="actual",
            )

        # Create features for the forecasting model
        features_df, net_load_profile = self.create_feature_df(weather_data, net_load_profile)

        # Create and train the forecasting model
        features_names = features_df.columns.tolist()
        all_gam_terms = (
            f(features_names.index("day_of_week")) +
            te(
                s(features_names.index("hour_of_day"), n_splines=5),
                s(features_names.index("temperature_2m"), n_splines=5),
            ) +
            te(
                s(features_names.index("hour_of_day"), n_splines=10, lam=0.3),
                s(features_names.index("global_tilted_irradiance"), n_splines=10, lam=0.15),
            ) +
            te(
                s(features_names.index("temperature_2m"), n_splines=10, lam=0.3),
                s(features_names.index("wind_speed_10m"), n_splines=10, lam=0.15),
            ))
        if self.use_lagged_power:
            all_gam_terms += s(features_names.index("lag_7_days"))

            # Cut the first 7 days, because of the 7d lag
            net_load_profile = net_load_profile[self.lagged_power:]
            features_df = features_df[self.lagged_power:]

        my_model = Gam(all_gam_terms)
        x = features_df.values
        y = net_load_profile.values
        history = my_model.train_model(x, y)
        predictions = my_model.predict(x)

        # Evaluate the model using nMAE
        nmae = self.calculate_nmae(net_load_profile.values, predictions)
        print(f"nMae: {nmae:.2f}%")

        # Plot actual vs predicted load profile
        plt.figure(figsize=(12, 6))
        plt.plot(net_load_profile.index, net_load_profile.values, label='Actual Load', color='blue')
        plt.plot(net_load_profile.index, predictions, label='Predicted Load', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Net Load (kW)')
        plt.title('Actual vs Predicted Net Load Profile')
        plt.legend()
        plt.grid()
        plt.show()

        # Save the model as a pickle file.
        if os.path.exists(self.model_filename):
            print(f"File {self.model_filename} already exists.")
        else:
            with open(self.model_filename, 'wb') as file:
                pickle.dump(my_model, file)

    def calculate_nmae(self, y_true: pd.Series | torch.Tensor, y_pred: pd.Series | torch.Tensor) -> float:
        loss_fn = torch.nn.L1Loss()
        reference = y_true.max() - y_true.min()
        mae = loss_fn(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32))
        n_mae = 100.0 * mae / reference
        return n_mae.item()

if __name__ == "__main__":
    forecasting_model = ForecastingModel()
    forecasting_model.train_model()
