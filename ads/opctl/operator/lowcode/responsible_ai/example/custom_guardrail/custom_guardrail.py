import pandas as pd
import evaluate
import datasets

_DESCRIPTION = "custom metric: calculate text length."
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2020}
}
"""
_KWARGS_DESCRIPTION = """
Args:
    `prediction`: a list of `str` for which the text length is calculated.

Returns:
    `text length` (`int`) : the length of the text.
Examples:
    >>> data = ["hello world"]
    >>> wordlength = evaluate.load("ads/opctl/operator/lowcode/responsible_ai/example/custom_guardrail/custom_guardrail.py", module_type="measurement")
    >>> results = wordlength.compute(predictions=data)
    >>> print(results)
    {'average_word_length': pd.DataFrame([[11, "hello world"]], columns["length", "text"])}
"""


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
            codebase_urls=["https://huggingface.co/spaces/evaluate-measurement/word_length"],
            reference_urls=[
                "https://www.nltk.org/api/nltk.tokenize.html",
            ],
        )

    def _compute(self, predictions: pd.Series, **kwargs):
        return {
            "custom_metric": pd.DataFrame(
                [[len(pred), pred] for pred in predictions], columns=["length", "text"]
            )
        }
