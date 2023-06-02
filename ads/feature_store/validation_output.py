import pandas as pd
from typing import Dict
from copy import deepcopy

from ads.feature_store.response.response_builder import ResponseBuilder
from ads.jobs.builders.base import Builder
from ads.common import utils


class ValidationOutput(ResponseBuilder):
    """
    Represents validation output results class after validation.
    """

    def to_pandas(self) -> pd.DataFrame:
        """
        Converts the validation output information to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The validation output information as a pandas DataFrame.
        """
        if self.content:
            profile_result = pd.json_normalize(self.content)
            return profile_result

    @property
    def kind(self) -> str:
        """
        Gets the kind of the validation output object.

        Returns
        -------
        str
            The kind of the validation output object, which is always "ValidationOutput".
        """
        return "validationoutput"
