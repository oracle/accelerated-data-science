import pandas as pd

class CustomGuardRail:

    def load(self, load_args: dict={}):
        return 

    def compute(self, evaluator, predictions: pd.Series, references: pd.Series=None, **kwargs):
        
        return {"custom_metric": pd.DataFrame([[len(pred), pred] for pred in predictions], columns=["length", 'predictions'])}

    @property
    def description(self):
        return "custom metric: calculate sentence length."