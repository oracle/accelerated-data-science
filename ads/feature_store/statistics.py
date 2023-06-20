import pandas as pd
from typing import Dict
from copy import deepcopy

from ads.feature_store.response.response_builder import ResponseBuilder
from ads.jobs.builders.base import Builder
from ads.common import utils


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
