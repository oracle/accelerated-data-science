import json
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
            validation_output_json = json.loads(self.content)
            profile_result = pd.json_normalize(
                validation_output_json.get("results")
            ).transpose()
            return profile_result

    def to_summary(self) -> pd.DataFrame:
        """
        Converts the validation output summary information to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The validation output summary information as a pandas DataFrame.
        """
        if self.content:
            validation_output_json = json.loads(self.content)
            profile_result = pd.json_normalize(validation_output_json).transpose()
            summary_df = profile_result.drop("results")
            return summary_df

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
