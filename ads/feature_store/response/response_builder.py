import pandas as pd
from typing import Dict
from copy import deepcopy
from ads.jobs.builders.base import Builder
from ads.common import utils


class ResponseBuilder(Builder):
    """
    Represents ResponseBuilder class to generate stats and validation reports for the feature store.
    """

    CONST_CONTENT = "content"

    def __init__(self, content: str, version: int = 1):
        """
        Initializes a new instance of the validation output class.

        Parameters
        ----------
        content : str
            The validation output information as a JSON string.
        """
        self.version = version
        super().__init__()
        self.set_spec(self.CONST_CONTENT, content)

    @property
    def content(self):
        """
        Gets the content as a JSON string.

        Returns
        -------
        str
            The validation output information as a JSON string.
        """
        return self.get_spec(self.CONST_CONTENT)

    def to_pandas(self) -> pd.DataFrame:
        """
        Converts the content to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The statistical information as a pandas DataFrame.
        """
        if self.content:
            profile_result = pd.read_json(self.content)
            return profile_result

    @property
    def kind(self) -> str:
        return "response_builder"

    def to_dict(self) -> Dict:
        spec = deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": utils.batch_convert_case(spec, "camel"),
        }
