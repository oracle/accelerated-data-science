import evaluate
import pandas as pd

class CustomGuardRail:

    def load(self, load_args: dict={}):
        return evaluate.load("honest", "en", **load_args)

    def compute(self, evaluator, predictions: pd.Series, references: pd.Series=None, **kwargs):
        return {"custom_metric": pd.DataFrame([[0]], columns=["custom_metric"])}