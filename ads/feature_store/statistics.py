from ads.feature_store.feature_stat import FeatureStatistics
from ads.feature_store.response.response_builder import ResponseBuilder
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

    def to_viz(self):
        if self.content is not None:
            stats: dict = json.loads(self.content)
            [
                FeatureStatistics.from_json(feature, stat).to_viz()
                for feature, stat in stats.items()
                if FeatureStatistics.from_json(feature, stat) is not None
            ]
