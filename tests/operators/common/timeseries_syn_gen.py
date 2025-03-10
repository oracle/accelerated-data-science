import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TimeSeriesGenerator:
    """
    A class to generate synthetic timeseries data with various features and target values.

    Attributes:
    - num_series: Number of different time series to generate.
    - num_points: Number of data points per time series.
    - start_date: Start date for the time series.
    - non_linear_func: Function to apply non-linear transformation to feature_3.
    - coeffs: Dictionary of coefficients for the features. Defaults to 1 if not provided.
    - freq: Frequency of the datetime column. Options: 'D' (daily), 'W' (weekly), 'M' (monthly), '2W' (bi-weekly), 'Y' (yearly), 'H' (hourly), 'T' (minutely).
    - freq_map: Mapping of frequency options to timedelta values.
    - static_1, static_2, static_3: Static features that remain constant for each series.
    - seasonality: Dictionary of seasonalities for the features. Defaults to predefined values if not provided.
    - trend_type: Type of trend ('linear', 'quadratic', 'exponential', 'logarithmic').
    - trend_direction: Direction of trend ('increasing', 'decreasing').
    """

    def __init__(
        self,
        num_series=10,
        num_points=100,
        start_date="2023-01-01",
        non_linear_func=None,
        coeffs=None,
        freq="D",
        seasonality=None,
        trend_type="linear",
        trend_direction="increasing",
        horizon=1,
        seed=42,
    ):
        """
        Initialize the TimeSeriesGenerator with the given parameters.
        """
        self.num_series = num_series
        self.num_points = num_points
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.non_linear_func = (
            non_linear_func if non_linear_func else lambda x: np.sin(x)
        )
        self.coeffs = (
            coeffs
            if coeffs
            else {
                "feature_1": 1,
                "feature_2": 1,
                "feature_3": 1,
                "static_1": 0.1,
                "static_2": 0.1,
                "static_3": 0.1,
            }
        )
        self.freq = freq
        self.freq_map = {
            "D": timedelta(days=1),
            "W": timedelta(weeks=1),
            "2W": timedelta(weeks=2),
            "M": timedelta(days=30),
            "Y": timedelta(days=365),
            "H": timedelta(hours=1),
            "T": timedelta(minutes=1),
        }
        self.static_1 = np.random.RandomState(seed).randint(0, 100, self.num_series)
        self.static_2 = np.random.RandomState(seed + 1).randint(0, 100, self.num_series)
        self.static_3 = np.random.RandomState(seed + 2).randint(0, 100, self.num_series)
        self.seasonality = (
            seasonality
            if seasonality
            else {"feature_1": 30, "feature_2": 30, "feature_3": 15}
        )
        self.trend_type = trend_type
        self.trend_direction = trend_direction
        self.trend_feature = self.generate_trend()
        self.horizon = max(1, horizon)  # Ensure horizon is at least 1
        self.seed = seed

    def generate_trend(self):
        """
        Generate a trend based on the specified type and direction.

        Returns:
        - Numpy array representing the trend.
        """
        t = np.arange(self.num_points)
        if self.trend_type == "linear":
            trend = t
        elif self.trend_type == "quadratic":
            trend = t**2
        elif self.trend_type == "exponential":
            trend = np.exp(t / self.num_points)
        elif self.trend_type == "logarithmic":
            trend = np.log(t + 1)
        else:
            trend = t

        if self.trend_direction == "decreasing":
            trend = -trend

        return trend / np.max(np.abs(trend))

    def generate_dates(self):
        """
        Generate a list of dates based on the start date and frequency.

        Returns:
        - List of datetime objects.
        """
        return [
            self.start_date + i * self.freq_map[self.freq]
            for i in range(self.num_points + self.horizon)
        ]

    def generate_features(self):
        """
        Generate random features for the time series with positive values and seasonality/trend.

        Returns:
        - Tuple of three numpy arrays representing the features.
        """
        t = np.arange(self.num_points + self.horizon)
        rng = np.random.RandomState(self.seed)
        feature_1 = np.abs(
            np.sin(2 * np.pi * t / self.seasonality["feature_1"])
            + rng.randn(self.num_points + self.horizon) * 0.1
        )
        feature_2 = np.abs(
            np.cos(2 * np.pi * t / self.seasonality["feature_2"])
            + rng.randn(self.num_points + self.horizon) * 0.1
        )
        feature_3 = np.abs(
            np.sin(2 * np.pi * t / self.seasonality["feature_3"])
            + rng.randn(self.num_points + self.horizon) * 0.1
        )
        fourier_1 = np.sin(2 * np.pi * t / 365.25)
        fourier_2 = np.cos(2 * np.pi * t / 365.25)
        return feature_1, feature_2, feature_3, fourier_1, fourier_2

    def calculate_target(
        self, feature_1, feature_2, feature_3, fourier_1, fourier_2, series_id
    ):
        """
        Calculate the target value based on the features and static values.

        Parameters:
        - feature_1, feature_2, feature_3, fourier_1, fourier_2: Numpy arrays representing the features.
        - series_id: Integer representing the series ID.

        Returns:
        - Numpy array representing the target values.
        """
        rng = np.random.RandomState(self.seed + series_id)
        noise = (
            rng.randn(self.num_points) * 5
        )  # Adding noise for more realistic variations
        return (
            self.coeffs.get("feature_1", 10) * feature_1
            + self.coeffs.get("feature_2", 10) * feature_2
            + self.non_linear_func(self.coeffs.get("feature_3", 10) * feature_3)
            + self.coeffs.get("static_1", 0.1) * self.static_1[series_id]
            + self.coeffs.get("static_2", 0.1) * self.static_2[series_id]
            + self.coeffs.get("static_3", 0.1) * self.static_3[series_id]
            + self.coeffs.get("fourier_1", 5) * fourier_1
            + self.coeffs.get("fourier_2", 5) * fourier_2
            + self.trend_feature
            + noise
        )

    def generate_series(self, series_id):
        """
        Generate a single time series with the given series ID.

        Parameters:
        - series_id: Integer representing the series ID.

        Returns:
        - DataFrame containing the generated time series data.
        """
        dates = self.generate_dates()
        feature_1, feature_2, feature_3, fourier_1, fourier_2 = self.generate_features()
        target = self.calculate_target(
            feature_1[: self.num_points],
            feature_2[: self.num_points],
            feature_3[: self.num_points],
            fourier_1[: self.num_points],
            fourier_2[: self.num_points],
            series_id,
        )

        data = {
            "series_id": [series_id] * (self.num_points + self.horizon),
            "ds": dates,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "fourier_1": fourier_1,
            "fourier_2": fourier_2,
            "static_1": [self.static_1[series_id]] * (self.num_points + self.horizon),
            "static_2": [self.static_2[series_id]] * (self.num_points + self.horizon),
            "static_3": [self.static_3[series_id]] * (self.num_points + self.horizon),
            "trend_feature": np.concatenate(
                [self.trend_feature, np.zeros(self.horizon)]
            ),
            "target": np.concatenate([target, np.zeros(self.horizon)]),
        }

        return pd.DataFrame(data)

    def generate_timeseries_data(self):
        """
        Generate the complete timeseries data for all series.

        Returns:
        - Tuple of two DataFrames: primary and additional.
        """
        series_list = [
            self.generate_series(series_id) for series_id in range(self.num_series)
        ]
        full_data = pd.concat(series_list, ignore_index=True)

        primary = (
            full_data.groupby("series_id")
            .apply(lambda df: df.iloc[: self.num_points])
            .reset_index(drop=True)[["series_id", "ds", "target"]]
        )
        additional = full_data.drop(columns=["target"])

        return primary, additional


if __name__ == "__main__":
    generator = TimeSeriesGenerator(
        non_linear_func=np.cos,
        coeffs={
            "feature_1": 2,
            "feature_2": 3,
            "feature_3": 0.5,
            "static_1": 0.1,
            "static_2": 0.1,
            "static_3": 0.1,
            "fourier_1": 0.3,
            "fourier_2": 0.3,
        },
        freq="T",
        trend_type="exponential",
        trend_direction="increasing",
    )
    primary, additional = generator.generate_timeseries_data()
    print(primary.tail(20), primary.shape)
    print(additional.tail(20), additional.shape)
