#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.common.data
"""
import os
import pandas as pd
import pytest
import unittest

from ads.common.data import ADSData

#
# run with:
#  python -m pytest -v -p no:warnings --cov-config=.coveragerc --cov=./ --cov-report html /home/datascience/advanced-ds/tests/unitary/test_common_data_ADSData.py
#
class ADSDataTest(unittest.TestCase):
    """
    Contains test cases for ads.common.data
    """

    data = pd.DataFrame(
        {
            "sepal_length": [
                5.0,
                5.0,
                4.4,
                5.5,
                5.5,
                5.1,
                6.9,
                6.5,
                5.2,
                6.1,
                5.4,
                6.3,
                7.3,
                6.7,
            ],
            "sepal_width": [
                3.6,
                3.4,
                2.9,
                4.2,
                3.5,
                3.8,
                3.1,
                2.8,
                2.7,
                2.8,
                3,
                2.9,
                2.9,
                2.5,
            ],
            "petal_width": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                1.5,
                1.5,
                1.4,
                1.2,
                1.5,
                1.8,
                1.8,
                1.8,
            ],
            "class": [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
                "virginica",
            ],
            "petal_length": [
                1.4,
                1.5,
                1.4,
                1.4,
                1.3,
                1.6,
                4.9,
                4.6,
                3.9,
                4.7,
                4.5,
                5.6,
                6.3,
                5.8,
            ],
        }
    )

    def test_ADSData_build_bad_input(self):
        """
        Test corner cases and error handling
        """
        bad_input = [(None, None), ("test", None), (None, "test")]
        for (X, y) in bad_input:
            with pytest.raises(ValueError):
                ADSData.build(X, y)

    def test_ADSData_build_valid_input_using_target_name(self):
        """
        Ensures build method takes pandas dataframe and target variable name
        """
        expected = ADSData.build(self.data, y="class")
        assert expected.X.columns.tolist() == [
            "sepal_length",
            "sepal_width",
            "petal_width",
            "petal_length",
        ]
        assert expected.X.shape == (14, 4)
        assert expected.y.name == "class"
        assert expected.y.shape == (14,)

    def test_ADSData_build_valid_input_using_target_vector(self):
        """
        Ensures build method takes training dataframe and labels
        """
        X = pd.DataFrame(
            {
                "c1": [1, 2, 3],
                "c2": [42.0, 3.14, 2.71],
                "c3": ["X", "Y", "Z"],
            }
        )
        y = pd.Series(["y", "y", "n"], name="target")
        expected = ADSData.build(X, y)
        assert expected.X.columns.tolist() == ["c1", "c2", "c3"]
        assert expected.X.shape == (3, 3)
        assert expected.y.name == "target"
        assert expected.y.shape == (3,)

    @pytest.mark.skipif("NoDependency" in os.environ, reason="skip for dependency test")
    def test_ADSData_build_valid_input_dask_dataframe(self):
        """
        Ensures build method takes dask dataframe
        """
        import dask

        X = dask.datasets.timeseries().drop("y", axis=1)
        y = dask.datasets.timeseries()["y"]
        expected = ADSData.build(X, y)
        assert sorted(expected.X.columns.tolist()) == sorted(["id", "name", "x"])
        assert expected.X.shape[0] == 2592000
        assert expected.X.shape[1] == 3
        assert expected.y.name == "y"
        assert expected.y.shape[0] == 2592000

    @pytest.mark.skip("api change. this test should be re-written.")
    def test_ADSData_build_with_data(self):
        """
        Ensures build method can take only the entire dataframe
        """
        expected = ADSData.build(X=None, y=None, data=self.data)
        assert expected is not None

    def test_ADSData_getattr(self):
        """
        Ensures _getattr_ method returns correct value
        """
        fake_data = ADSData(self.data, y="class")
        assert fake_data.dataset_type == None
        assert fake_data.X.shape == (14, 5)

    def test_ADSData_build_X_str_invalid(self):
        """
        Test corner case of passing in only str in build method
        """
        with pytest.raises(ValueError):
            expected = ADSData.build(X="test", y="test")
