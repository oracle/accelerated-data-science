#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
from ads.common.utils import _is_dask_dataframe, _is_dask_series


class ADSData(object):
    def __init__(self, X=None, y=None, name="", dataset_type=None):
        r"""
        This class wraps the input dataframe to various models, evaluation, and explanation frameworks.
        It's primary purpose is to hold any metadata relevant to these tasks. This can include it's:

        - X - the independent variables as some dataframe-like structure,
        - y - the dependent variable or target column as some array-like structure,
        - name - a string to name the data for user convenience,
        - dataset_type - the type of the X value.

        As part of this initiative, ADSData knows how to turn itself into an onnxruntime compatible data
        structure with the method .to_onnxrt(), which takes and onnx session as input.

        Parameters
        ----------
        X : Union[pandas.DataFrame, dask.DataFrame, numpy.ndarray, scipy.sparse.csr.csr_matrix]
            If str, URI for the dataset. The dataset could be read from local or network file system, hdfs, s3 and gcs
            Should be none if X_train, y_train, X_test, Y_test are provided
        y: Union[str, pandas.DataFrame, dask.DataFrame, pandas.Series, dask.Series, numpy.ndarray]
            If str, name of the target in X, otherwise series of labels corresponding to X
        name: str, optional
            Name to identify this data
        dataset_type: ADSDataset optional
            When this value is available, would be used to evaluate the ads task type
        kwargs:
            Additional keyword arguments that would be passed to the underlying Pandas read API.
        """
        self.X = X
        self.y = y
        self.name = name
        self.dataset_type = dataset_type

    @staticmethod
    def build(X=None, y=None, name="", dataset_type=None, **kwargs):
        r"""
        Returns an ADSData object built from the (source, target) or (X,y)

        Parameters
        ----------
        X : Union[pandas.DataFrame, dask.DataFrame, numpy.ndarray, scipy.sparse.csr.csr_matrix]
            If str, URI for the dataset. The dataset could be read from local or network file system, hdfs, s3 and gcs
            Should be none if X_train, y_train, X_test, Y_test are provided
        y: Union[str, pandas.DataFrame, dask.DataFrame, pandas.Series, dask.Series, numpy.ndarray]
            If str, name of the target in X, otherwise series of labels corresponding to X
        name: str, optional
            Name to identify this data
        dataset_type: ADSDataset, optional
            When this value is available, would be used to evaluate the ads task
            type
        kwargs:
            Additional keyword arguments that would be passed to the underlying Pandas read API.

        Returns
        -------
        ads_data: ads.common.data.ADSData
            A built ADSData object

        Examples
        --------
        >>> data = open_csv("my.csv")

        >>> data_ads = ADSData(data, 'target').build(data, 'target')
        """
        if X is None or y is None:
            raise ValueError("Both X and y are required.")
        if _is_dask_dataframe(X):
            X = X.compute()
        if _is_dask_series(y):
            y = y.compute()
        if dataset_type is None:
            dataset_type = type(X)
        if isinstance(y, str):
            try:
                return ADSData(
                    X.drop(y, axis=1), X[y], name=name, dataset_type=dataset_type
                )
            except AttributeError:
                raise ValueError(
                    "If y is a string, then X must be a pandas or dask dataframe"
                )
        else:
            return ADSData(X, y, name=name, dataset_type=dataset_type)

    def __repr__(self):
        return "%sShape of X:%s\nShape of y:%s" % (
            self.name + "\n",
            str(self.X.shape),
            str(self.y.shape),
        )

    def to_onnxrt(
        self, sess, idx_range=None, model=None, impute_values={}, **kwargs
    ):  # pragma: no cover
        r"""
        Returns itself formatted as an input for the onnxruntime session inputs passed in.

        Parameters
        ----------
        sess: Session
            The session object
        idx_range: Range
            The range of inputs to convert to onnx
        model: SupportedModel
            A model that supports being serialized for the onnx runtime.
        kwargs: additional keyword arguments

            - sess_inputs - Pass in the output from onnxruntime.InferenceSession("model.onnx").get_inputs()
            - input_dtypes (list) - If sess_inputs cannot be passed in, pass in the numpy dtypes of each input
            - input_shapes (list) - If sess_inputs cannot be passed in, pass in the shape of each input
            - input_names (list) -If sess_inputs cannot be passed in, pass in the name of each input

        Returns
        -------
        ort: Array
            array of inputs formatted for the given session.
        """
        if model._underlying_model in ["torch"]:
            sess_inputs = sess.get_inputs()
            in_shape, in_name, in_type = [], [], []
            for i, ftr in enumerate(sess_inputs):
                in_type.append(ftr.type)
                in_shape.append(ftr.shape)
                in_name.append(ftr.name)
            ret = {}
            for i, name in enumerate(in_name):
                idx_range = (0, len(self.X)) if idx_range is None else idx_range
                batch_size = idx_range[1] - idx_range[0]
                ret[name] = (
                    self.X[:batch_size]
                    .reshape([batch_size] + list(self.X[:1].shape))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            return ret
        elif model._underlying_model in ["automl"]:
            X_trans = model._onnx_data_transformer(
                X=self.X, impute_values=impute_values
            )
            inputs = {}
            for idx, c in enumerate(X_trans.columns):
                inputs[sess.get_inputs()[idx].name] = (
                    X_trans[c]
                    .values.reshape((X_trans.shape[0], 1))
                    .astype(X_trans.dtypes[idx])
                )
            return inputs
        elif model._underlying_model in ["lightgbm", "xgboost", "sklearn"]:
            idx_range = (0, len(self.X)) if idx_range is None else idx_range
            inputs = []
            for name, row in self.X[idx_range[0] : idx_range[1]].iterrows():
                inputs.append(list(row))
            return {"input": inputs}
