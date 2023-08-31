import pandas as pd
import evaluate
import datasets

_DESCRIPTION = "custom metric: calculate sentence length."
_CITATION = ""
_KWARGS_DESCRIPTION = ""


class CustomGuardRail(evaluate.Measurement):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=[""],
            reference_urls=[
                "",
            ],
        )

    def _compute(self, predictions: pd.Series, **kwargs):
        return {
            "custom_metric": pd.DataFrame(
                [[len(pred), pred] for pred in predictions], columns=["length", "text"]
            )
        }
