#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.explanations import MLXGlobalExplainer, MLXLocalExplainer
from ads.explanations.mlx_whatif_explainer import MLXWhatIfExplainer
from ads.explanations.mlx_interface import _reset_index
from ads.common.decorator.deprecate import deprecated


class ADSExplainer(object):
    """
    Main ADS Explainer class for configuring and managing local and global
    explanation objects.
    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self, test_data, model, training_data=None):
        """
        Creates an ads explainer object.

        Parameters
        ----------
        test_data : ads.common.data.ADSData
            The test data for exaplanation built with `ADSData.build()`.
        model : ads.common.model.ADSModel
            The model used for explanation built with `ADSModel.from_estimator()`.
            Maximum length of the list is 3.
        training_data : ads.common.data.ADSData instance, optional
            Training data using `ADSData.build()`.
            Using `ADSData.build()`.

        Examples
        --------
        train, test = ds.train_test_split()

        model = MyModelClass.train(train)

        explainer = ADSExplainer(test, model)
        """

        self.X_train = None
        self.y_train = None
        if training_data is not None:
            self.X_train, self.y_train = training_data.X, training_data.y
        self.X_test, self.y_test = test_data.X, test_data.y
        self.model = model

    def global_explanation(self, provider=MLXGlobalExplainer()):
        """
        Initializes the given global explainer provider and returns it. The returned
        explainer will have the following properties set:

        * **est**: Estimator in the ADSModel
        * **class_names** (`Iterable`): Target classes. None for regression
        * **X_test** (`pandas.DataFrame`): Test data to explain.
        * **y_test** (`pandas.Series`): Labels for test data.
        * **X_train** (`pandas.DataFrame`, optional): Training data to build explainer.
        * **y_train** (`pandas.Series`, optional) Labels for training data.

        Parameters
        ----------
        provider : instance of `GlobalExplainer`, optional
            The explainer instance that is set up. Defaults to `MLXGlobalExplainer`.
            The properties will be set on the object

        Return
        ------
        `GlobalExplainer` object
            Modified instance of `GlobalExplainer` implementation passed into function

        Examples
        --------
        explainer = ADSExplainer(test, model)

        global_explanation = explainer.global_explanation()
        """

        provider.setup(self.model, self.X_test, self.y_test, self.X_train, self.y_train)
        return provider

    def local_explanation(self, provider=MLXLocalExplainer()):
        """Initializes the given local explainer provider and returns it.

        The returned `LocalExplainer` instance has the following properties set:

        * **est**: Estimator in the ADSModel.
        * **class_names** (`Iterable`): Target classes. None for regression.
        * **X_test** (`pandas.DataFrame`): Test data to explain.
        * **y_test** (`pandas.Series`): Labels for test data.
        * **X_train** (`pandas.DataFrame`, optional): Training data to build explainer.
        * **y_train** (`pandas.Series`, optional): Labels for training data.

        Parameters
        ----------
        provider : instance of `LocalExplainer`, optional
            The explainer instance that is set up. Defaults to `MLXLocalExplainer`.

        Return
        ------
        `LocalExplainer` object
            Modified instance of `LocalExplainer` implementation passed into function.

        Examples
        --------
        explainer = ADSExplainer(test, model)

        local_explanation = explainer.local_explanation()
        """

        provider.setup(self.model, self.X_test, self.y_test, self.X_train, self.y_train)
        return provider

    def whatif_explanation(self, provider=MLXWhatIfExplainer()):
        """Initializes the given mlx explorer provider and returns it.

        The returned `MLXWhatIfExplainer` instance would have the following properties set:

        * **est**: Estimator in the ADSModel.
        * **class_names** (Iterable): Target classes. None for regression.
        * **X_test** (pandas.DataFrame): Test data to explain.
        * **y_test** (pandas.Series): Labels for test data.
        * **X_train** (pandas.DataFrame, optional): Training data to build explainer.
        * **y_train** (pandas.Series, optional): Labels for training data.

        Parameters
        ----------
        provider : Instance of `MLXWhatIfExplainer` implementation, optional.
            Defaults to `MLXWhatIfExplainer`

        Return
        ------
        `MLXWhatIfExplainer` object
            Configured explanation provider.

        Examples
        --------
        explainer = ADSExplainer(test, model)

        local_explanation = explainer.explore_sample()
        """

        if self.X_train is None:
            provider.setup(
                model=self.model,
                X_test=_reset_index(self.X_test),
                y_test=_reset_index(self.y_test),
                X_train=self.X_train,
                y_train=self.y_train,
            )
        else:
            provider.setup(
                model=self.model,
                X_test=_reset_index(self.X_test),
                y_test=_reset_index(self.y_test),
                X_train=_reset_index(self.X_train),
                y_train=_reset_index(self.y_train),
            )
        return provider

    def show_in_notebook(self, provider=MLXGlobalExplainer()):
        """Generates a global or local explanation based on the default values and
        returns a visualization for the explanation.

        Parameters
        ----------
        provider : Instance of `GlobalExplainer` or `LocalExplainer` implementation, optional
            Configures the type of explanation to generate and visualize (local or global).
            Default is `MLXGlobalExplainer()`. Currently does not support `MLXWhatIfExplainer`.

        Return
        ------
        `None`
            Nothing


        Raises
        ------
        TypeError
            If `provider` is an instance of `MLXWhatIfExplainer`
        """

        if isinstance(provider, MLXWhatIfExplainer):
            raise TypeError("MLXWhatIfExplainer does not support show_in_notebook.")
        self.global_explanation(provider=provider).show_in_notebook()
