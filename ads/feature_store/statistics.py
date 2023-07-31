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

    def to_viz(self, feature=None):
        """
           Converts the content to a matplotlib plot.

           Returns
           -------
          None
        """
        if self.content:
            feature_metrics = json.loads(self.content)
            if feature is not None:
                fs_utils.plot_table(feature, self.extract_scaler_metrics(feature_metrics.get(feature)))
            else:
                for feature, metrics in feature_metrics.items():
                    fs_utils.plot_table(feature, self.extract_scaler_metrics(metrics))

    def extract_scaler_metrics(self, metrics):
        scaler_metrics = []
        for metric_name, data in metrics.items():
            if 'value' in data and isinstance(data.get('value', 0), (float, int, bool)):
                scaler_metrics.append((metric_name, data.get('value')))
        return scaler_metrics
