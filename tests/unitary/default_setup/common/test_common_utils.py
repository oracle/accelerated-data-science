#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pathlib
import random
import shutil
import sys
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split

import ads
from ads.common import utils
from ads.common.auth import AuthState, AuthType
from ads.common.utils import (
    JsonConverter,
    copy_file,
    copy_from_uri,
    extract_region,
    folder_size,
    human_size,
    remove_file,
)

DEFAULT_SIGNER_CONF = {"config": {}}


class TestCommonUtils:
    """Contains test cases for common.utils"""

    OCI_CONFIG_LOCATION = "/home/datascience/.oci/config"
    OCI_CONFIG_DIR = "/home/datascience/.oci"
    OCI_CONFIG_PROFILE = "Default"

    json_conv = JsonConverter()

    tmp_model_dir = "/tmp/model"

    random_seed = 42

    @patch.dict(os.environ, {"OCI_CONFIG_LOCATION": "/home/datascience/.oci/config"})
    def test_get_oci_config(self):
        """Test get_oci_config returns correct the OCI config location, and the OCI config profile."""
        oci_config_location, oci_config_profile = utils.get_oci_config()
        assert self.OCI_CONFIG_LOCATION == oci_config_location

    @patch.dict(os.environ, {"OCI_CONFIG_DIR": "/home/datascience/.oci"})
    def test_oci_key_config_location(self):
        """Test oci_key_location and oci_config_file return correct OCI key and config location."""
        key_loc = utils.oci_key_location()
        assert self.OCI_CONFIG_DIR == key_loc

        config_loc = utils.oci_config_file()
        assert self.OCI_CONFIG_LOCATION == config_loc

    @patch.dict(os.environ, {"OCI_CONFIG_PROFILE": "Default"})
    def test_oci_config_profile(self):
        profile = utils.oci_config_profile()
        assert self.OCI_CONFIG_PROFILE == profile

    @patch("os.path.exists")
    def test_set_auth_non_default_config_and_profile(self, mock_exists):
        mock_exists.return_value = True
        ads.set_auth(
            auth=AuthType.API_KEY,
            oci_config_location="~/.oci_test/config",
            profile="TEST_PROFILE",
        )

        assert AuthState().oci_config_path == "~/.oci_test/config"
        assert AuthState().oci_key_profile == "TEST_PROFILE"
        ads.set_auth(auth=AuthType.API_KEY)

    def test_numeric_pandas_dtypes(self):
        """Test numeric_pandas_dtypes returns correct a list of the "numeric" pandas data types"""
        expected_res = ["int16", "int32", "int64", "float16", "float32", "float64"]
        res = utils.numeric_pandas_dtypes()
        assert expected_res == res

    def test_random_valid_ocid(self):
        """Test generating a random valid ocid."""
        ocid = utils.random_valid_ocid()
        prefix = ocid.rsplit(".", 1)[0]
        assert prefix == "ocid1.dataflowapplication.oc1"

    def test_common_utils(self):
        """Test some method in common.utils. copied from test_automl_plotting.py"""
        styles = utils.get_dataframe_styles()

        bootstrap = utils.get_bootstrap_styles()

        h_text = utils.highlight_text("some text!")

        my_div = "a fake div string!"
        hhorz = utils.horizontal_scrollable_div(my_div)

        is_notebook = utils.is_notebook()

        e_string = utils.ellipsis_strings("this is sample text", n=24)
        e_string = utils.ellipsis_strings("this is sample text", n=4)
        e_string = utils.ellipsis_strings(["This", "is", "sample", "text"], n=3)

        a_df = pd.DataFrame(["This", "is", "sample", "text"])
        e_string = utils.ellipsis_strings(a_df, n=3)

        no_ = utils.first_not_none("")

        string_no_space = utils.replace_spaces("This is a string")
        assert string_no_space == [
            "T",
            "h",
            "i",
            "s",
            "_",
            "i",
            "s",
            "_",
            "a",
            "_",
            "s",
            "t",
            "r",
            "i",
            "n",
            "g",
        ]
        data = {
            "col2": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            "col5": [
                "x",
                "y",
                "y",
                "x",
                "y",
                "x",
                "x",
                "x",
                "y",
                "y",
                "x",
                "x",
                "x",
                "x",
                "x",
                "y",
                "x",
                "x",
                "x",
                "y",
            ],
        }
        df = pd.DataFrame(data=data)
        the_x = df["col5"]
        the_y = df["col2"]
        X_train, X_test, y_train, y_test = utils.split_data(
            the_x, the_y, random_state=42, test_size=0.2
        )

        nested_dict = {"dictA": {"key_1": "value_1"}, "dictB": {"key_2": "value_2"}}
        flat_dict = utils.flatten(nested_dict)

    def test_json_converter(self):
        """Test converting different types of object to JSON."""
        timestamp = pd.Timestamp("2017-01-01T12")
        json = self.json_conv.default(timestamp)
        assert json == "2017-01-01 12:00:00"

        series = pd.Series([1, 2, 3])
        json = self.json_conv.default(series)
        assert json.tolist() == [1, 2, 3]

        dataframe = pd.DataFrame(
            [["a", "b"], ["c", "d"]],
            index=["row 1", "row 2"],
            columns=["col 1", "col 2"],
        )
        json = self.json_conv.default(dataframe)
        assert json["col 1"]["row 1"] == "a"

        ndarr = np.array([[1, 2], [3, 4]])
        json = self.json_conv.default(ndarr)
        assert json == [[1, 2], [3, 4]]

        int_a = np.int64(10)
        json = self.json_conv.default(int_a)
        assert json == 10

        float_b = np.float_(10.0)
        json = self.json_conv.default(float_b)
        assert json == 10.0

    def test_wrap_lines(self):
        """Test wrapping the elements of iterable into multi line string of fixed width"""
        line = []
        expected = ""
        out = utils.wrap_lines(line)
        assert out == expected

    def test_requirements_pipeline_randomforestclassifier(self):
        """Test extract_lib_dependencies_from_model and generate_requirement_file"""
        if os.path.exists(self.tmp_model_dir):
            shutil.rmtree(self.tmp_model_dir)
        os.makedirs(self.tmp_model_dir)

        from sklearn.ensemble import RandomForestClassifier

        from ads.common.model import ADSModel
        from ads.dataset.dataset_browser import DatasetBrowser

        iris = datasets.load_iris(as_frame=True)
        X, y = iris["data"], iris["target"]
        X, y = iris["data"], iris["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = RandomForestClassifier().fit(X_train, y_train)
        X_sample = X_train.head(3)
        y_sample = y_train.head(3)
        rf_model = ADSModel.from_estimator(clf)

        reqs = utils.extract_lib_dependencies_from_model(rf_model)
        assert "automl" not in reqs
        assert "ads" not in reqs
        assert "scikit-learn" in reqs

        utils.generate_requirement_file(reqs, self.tmp_model_dir)
        assert os.path.exists(
            os.path.join(self.tmp_model_dir, "requirements.txt")
        ), "requirements.txt does not exist"

        shutil.rmtree(self.tmp_model_dir)

    def test_copy_from_uri_fail(self):
        """Ensures copying fails in case of destination folder/file already exists and
        `force_overwrite` flag is not set to True.
        """
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            tempdir = Path(temp_dir)
            file_name = tempdir / "test.txt"
            file_name.write_text("bla bla bla")
            with pytest.raises(
                ValueError, match="The destination path already exists."
            ):
                copy_from_uri(uri="fake_uri", to_path=temp_dir)

    @pytest.mark.parametrize(
        "input_params, expected_result",
        [
            (
                {
                    "uri": "./test_files/archive/1.txt",
                    "to_path": "1.txt",
                    "unpack": False,
                    "force_overwrite": True,
                    "auth": None,
                },
                ["./1.txt"],
            ),
            (
                {
                    "uri": "./test_files/archive/",
                    "to_path": "./destination/",
                    "unpack": False,
                    "force_overwrite": True,
                    "auth": DEFAULT_SIGNER_CONF,
                },
                ["destination/1.txt", "destination/tmp1", "destination/tmp1/2.txt"],
            ),
            (
                {
                    "uri": "./test_files/archive.zip",
                    "to_path": "./destination/",
                    "unpack": True,
                    "force_overwrite": True,
                    "auth": DEFAULT_SIGNER_CONF,
                },
                ["destination/1.txt", "destination/tmp1", "destination/tmp1/2.txt"],
            ),
        ],
    )
    @patch("ads.common.auth.default_signer")
    def test_copy_from_uri(self, mock_default_signer, input_params, expected_result):
        """Tests copying file(s) to local path."""
        ignore = ("__pycache__", ".DS_Store")
        mock_default_signer.return_value = DEFAULT_SIGNER_CONF
        uri = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), input_params["uri"]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            to_path = os.path.join(temp_dir, input_params["to_path"])
            copy_from_uri(
                uri=uri,
                to_path=to_path,
                unpack=input_params["unpack"],
                auth=input_params["auth"],
            )
            if not input_params["auth"]:
                mock_default_signer.assert_called()

            file_paths = []
            walk_folder = to_path
            if pathlib.Path(to_path).suffix:
                walk_folder = os.path.dirname(os.path.join(temp_dir, ""))

            for dir_path, dirnames, filenames in os.walk(walk_folder):
                dir_relative_path = os.path.relpath(dir_path, temp_dir)
                file_paths += [
                    os.path.join(dir_relative_path, file) for file in filenames
                ]
                file_paths += [os.path.join(dir_relative_path, dir) for dir in dirnames]

            file_paths = [
                file_path
                for file_path in file_paths
                if not any((ig in file_path for ig in ignore))
            ]
            assert file_paths == expected_result

    @patch("ads.common.auth.default_signer")
    def test_copy_file(self, mock_default_signer):
        """Ensures copying fails in case of destination file already exists and
        `force_overwrite` flag is not set to True.
        """
        mock_default_signer.return_value = DEFAULT_SIGNER_CONF
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileExistsError):
                copy_file(uri_src="fake_uri", uri_dst=temp_dir)

    @pytest.mark.parametrize(
        "input_params, expected_result",
        [
            (
                {
                    "uri_src": "test_files/archive/1.txt",
                    "uri_dst": "./tmp/",
                    "force_overwrite": True,
                    "auth": None,
                },
                "tmp/1.txt",
            ),
            (
                {
                    "uri_src": "test_files/archive/1.txt",
                    "uri_dst": "1.txt",
                    "force_overwrite": True,
                    "auth": DEFAULT_SIGNER_CONF,
                },
                "1.txt",
            ),
            (
                {
                    "uri_src": "test_files/archive/1.txt",
                    "uri_dst": "1.txt",
                    "force_overwrite": True,
                    "auth": DEFAULT_SIGNER_CONF,
                    "chunk_size": 10,
                },
                "1.txt",
            ),
        ],
    )
    @patch("ads.common.auth.default_signer")
    def test_copy_file(self, mock_default_signer, input_params, expected_result):
        """Tests copying the file `uri_src` to the file or directory `uri_dst`."""
        mock_default_signer.return_value = DEFAULT_SIGNER_CONF
        uri_src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), input_params["uri_src"]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            uri_dst = os.path.join(temp_dir, input_params["uri_dst"])
            result_file_name = copy_file(
                uri_src=uri_src,
                uri_dst=uri_dst,
                auth=input_params["auth"],
                force_overwrite=input_params["force_overwrite"],
                chunk_size=input_params.get("chunk_size"),
            )
            if not input_params["auth"]:
                mock_default_signer.assert_called()

            assert result_file_name.endswith(expected_result)
            assert os.path.exists(result_file_name)

    @patch("ads.common.auth.default_signer")
    def test_remove_file_fail(self, mock_default_signer):
        """Ensures removing file fails in case of incorrect input parameters."""
        mock_default_signer.return_value = DEFAULT_SIGNER_CONF
        with pytest.raises(FileNotFoundError):
            remove_file("fake_file_path")

    @patch("ads.common.auth.default_signer")
    def test_remove_file(self, mock_default_signer):
        """Ensures file can be successfully removed from."""
        mock_default_signer.return_value = DEFAULT_SIGNER_CONF

        with tempfile.TemporaryDirectory() as temp_dir:
            uri_src = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_files/archive/1.txt",
            )
            uri_dst = os.path.join(temp_dir, "1.txt")
            result_file_name = copy_file(
                uri_src=uri_src,
                uri_dst=uri_dst,
            )
            assert os.path.exists(result_file_name)
            remove_file(result_file_name)
            assert not os.path.exists(result_file_name)

    def test_folder_size(self):
        """Tests calculating a size of the folder."""
        expected_result = 12  # in bytes
        uri_src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_files/archive/tmp1"
        )
        test_result = folder_size(uri_src)
        assert test_result == expected_result

    @pytest.mark.parametrize(
        "input_value,result_value",
        [
            (100, "100.0B"),
            (1024, "1.0KB"),
            (12345423, "11.77MB"),
            (2147483648, "2.0GB"),
        ],
    )
    def test_human_size(self, input_value, result_value):
        """Tests converting bytes size to a string representation."""
        assert human_size(input_value) == result_value

    def test_is_notebook_with_ipython_uninstalled(self):
        """
        negative test for ipython
        """
        with patch.dict(sys.modules, {"IPython": None}):
            with pytest.raises(ModuleNotFoundError):
                utils.print_user_message("test failed")

    def test_get_random_name_for_resource(self):
        """Test generator of randomly generated easy to remember name for oci resources - returns words and timestamp."""
        random.seed(self.random_seed)
        generated_random_name = utils.get_random_name_for_resource()
        assert generated_random_name.split("-")[0] == "delightful"
        assert generated_random_name.split("-")[1] == "donkey"

        try:
            datetime.strptime(generated_random_name[-19:], "%Y-%m-%d-%H:%M.%S")
        except ValueError:
            assert False

    @pytest.mark.parametrize(
        "input_params, expected_result",
        [
            (
                {"auth": None},
                "default_signer_region",
            ),
            (
                {
                    "auth": {"config": {}, "signer": MagicMock(region=None)},
                },
                "region_from_metadata",
            ),
            (
                {
                    "auth": {
                        "config": {"region": "config_region"},
                        "signer": MagicMock(),
                    },
                },
                "config_region",
            ),
            (
                {
                    "auth": {"config": {}, "signer": MagicMock(region="signer_region")},
                },
                "signer_region",
            ),
        ],
    )
    @patch(
        "ads.config.OCI_REGION_METADATA", '{"regionIdentifier":"region_from_metadata"}'
    )
    def test_extract_region(self, input_params, expected_result):
        """Ensures that a region can be successfully extracted from the env variables or signer."""
        with patch(
            "ads.common.auth.default_signer",
            return_value={"config": {"region": "default_signer_region"}},
        ):
            assert extract_region(input_params["auth"]) == expected_result
