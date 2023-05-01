from typing import Tuple
import pandas as pd
from ads.dataset import progress
from ads.dataset.dataset import ADSDataset
from ads.dataset.dataset_with_target import ADSDatasetWithTarget

@pd.api.extensions.register_dataframe_accessor("ads")
class ADSDatasetAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
