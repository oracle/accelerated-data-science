import pandas as pd
from ads.opctl.operator.lowcode.responsible_ai.guardrail.guardrail import BaseGuardRail

class CustomGuardRail(BaseGuardRail):

    def load(self, load_args: dict={}):
        return 

    def compute(self, predictions: pd.Series, references: pd.Series=None, **kwargs):
        
        return {"custom_metric": pd.DataFrame([[len(pred), pred] for pred in predictions], columns=["length", 'predictions'])}

    @property
    def description(self):
        return "custom metric: calculate sentence length."