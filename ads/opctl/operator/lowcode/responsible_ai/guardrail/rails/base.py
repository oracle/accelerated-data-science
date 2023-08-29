from abc import ABC, abstractmethod, abstractproperty
from typing import Union
import pandas as pd


class BaseGuardRail(ABC):
    def __init__(self, config: dict, **kwargs):
        self.evaluator = self.load(config)
        self.score = None

    def load(self):
        pass

    @abstractmethod
    def compute(
        self,
        predictions: Union[pd.Series, list],
        **kwargs: dict,
    ):
        pass

    @property
    def description(self):
        return (
            self.evaluator.description
            if self.evaluator is not None and hasattr(self.evaluator, "description")
            else ""
        )

    @property
    def homepage(self):
        return (
            self.evaluator.homepage
            if self.evaluator is not None and hasattr(self.evaluator, "homepage")
            else ""
        )
