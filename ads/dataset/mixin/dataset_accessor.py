#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Tuple
import pandas as pd
from ads.dataset import progress

from ads.type_discovery.typed_feature import TypedFeature

class ADSDatasetAccessMixin:

    def dataset(
        self,
        sampled_df=None,
        shape=None,
        name="",
        description=None,
        type_discovery=True,
        types={},
        metadata=None,
        progress=progress.DummyProgressBar(),
        transformer_pipeline=None,
        interactive=False,
        **kwargs,
    ):
        """Converts pandas DataFrame into ADS Dataset.

        Parameters
        ----------
        sampled_df: pandas.DataFrame, optional
            The sampled pandas DataFrame. Defaults to None.
        shape: Tuple, optional
            The shape of pandas DataFrame. Defaults to None.
        name: str, optional
            The name of ADS Dataset. Defaults to "".
        description: str, optional
            Text describing the dataset. Defaults to "".
        type_discovery: bool. optional
            If false, the data types of the dataframe are used as such.
            By default, the dataframe columns are associated with the best suited data types. Associating the features
            with the disovered datatypes would impact visualizations and model prediction. Defaults to True.
        types: dict, optional
            Dictionary of <feature_name> : <data_type> to override the data type of features. Defaults to {}.
        metadata: dict, optional
            The metadata of ADS Dataset. Defaults to None.
        progress: dataset.progress.ProgressBar, optional
            The progress bar for ADS Dataset. Defaults to progress.DummyProgressBar()
        transformer_pipeline: datasets.pipeline.TransformerPipeline, optional
            A pipeline of transformations done outside the sdk and need to be applied at the time of scoring
        kwargs: additional keyword arguments that would be passed to underlying dataframe read API
            based on the format of the dataset

        Returns
        -------
        ADSDataset: 
            An instance of ADSDataset

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.read_csv(<path_to_csv>)
        >>> ds = df.ads.dataset()
        """
        from ads.dataset.dataset import ADSDataset

        return ADSDataset.from_dataframe(
            df=self._obj,
            sampled_df=sampled_df,
            shape=shape,
            name=name,
            description=description,
            type_discovery=type_discovery,
            types=types,
            metadata=metadata,
            progress=progress,
            transformer_pipeline=transformer_pipeline,
            interactive=interactive,
            **kwargs
        )

    def dataset_with_target(
        self, 
        target: str,
        sampled_df: pd.DataFrame = None,
        shape: Tuple[int, int] = None,
        target_type: TypedFeature = None,
        positive_class=None,
        **kwargs,
    ):
        """Converts pandas DataFrame into ADS Dataset with target.

        Parameters
        ----------
        target: str, optional
            Name of the target in dataset.
            If set an ADSDatasetWithTarget object is returned, otherwise an ADSDataset object is returned which can be
            used to understand the dataset through visualizations
        sampled_df: pandas.DataFrame, optional
            The sampled pandas DataFrame. Defaults to None.
        shape: Tuple, optional 
            The shape of pandas DataFrame. Defaults to None.
        target_type: TypedFeature, optional
            The target type of ADS Dataset. Defaults to None.
        positive_class: Any, optional
            Label in target for binary classification problems which should be identified as positive for modeling.
            By default, the first unique value is considered as the positive label.
        kwargs: additional keyword arguments that would be passed to underlying dataframe read API
            based on the format of the dataset

        Returns
        -------
        ADSDatasetWithTarget: 
            An instance of ADSDatasetWithTarget

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.read_csv(<path_to_csv>)
        >>> ds = df.ads.dataset_with_target(target="target")
        """
        from ads.dataset.dataset_with_target import ADSDatasetWithTarget

        return ADSDatasetWithTarget.from_dataframe(
            df=self._obj,
            target=target,
            sampled_df=sampled_df,
            shape=shape,
            target_type=target_type,
            positive_class=positive_class,
            **kwargs
        )
