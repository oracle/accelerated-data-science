import pandas as pd

from ads.feature_store.response.response_builder import ResponseBuilder
from ads.feature_store.common.utils import utility as fs_utils
import json


class Statistics(ResponseBuilder):
    """
    Represents statistical information.
    """

    @property
    def kind(self) -> str:
        """
        Gets the kind of the statistics object.

        Returns
        -------
        str
            The kind of the statistics object, which is always "statistics".
        """
        return "statistics"

    def to_viz(self, features=None):
        """
         Converts the content to a matplotlib plot.

         Returns
         -------
        None
        """
        if self.content:
            statistics_payload = json.loads(self.content)
            categorical_features, numerical_features = fs_utils.extract_scaler_metrics(
                statistics_payload, features
            )

            if categorical_features:
                fs_utils.plot_table(categorical_features)

            if numerical_features:
                fs_utils.plot_table(numerical_features)
