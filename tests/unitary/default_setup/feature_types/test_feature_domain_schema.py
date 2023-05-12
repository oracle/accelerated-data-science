#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ads.common import logger
from ads.common.data import ADSData
from ads.common.model_artifact import ModelArtifact
from ads.feature_engineering.schema import (
    DEFAULT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Attribute,
    Expression,
    JsonSchemaLoader,
    Schema,
    YamlSchemaLoader,
    SchemaSizeTooLarge,
)
from sklearn.datasets import load_iris


@patch("ads.model.common.utils.fetch_manifest_from_conda_location")
def get_model_artifact_instance(model_dir, mock_fetch_manifest_from_conda_location):
    manifest = {
        "pack_path": "pack_path: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/pyspark/1.0/pyspv10",
        "python": "3.6",
        "slug": "pyspv10",
        "type": "data_science",
        "version": "1.0",
        "arch_type": "CPU",
        "manifest_version": "1.0",
        "name": "pyspark",
    }
    mock_fetch_manifest_from_conda_location.return_value = manifest
    model_artifact = ModelArtifact(
        model_dir, reload=False, create=True, ignore_deployment_error=True
    )
    return model_artifact


class TestFeatureDomainSchema:
    ### CreditCard
    visa = [
        "4532640527811543",
        "4556929308150929",
        "4539944650919740",
        "4485348152450846",
        "4556593717607190",
    ]
    mastercard = [
        "5334180299390324",
        "5111466404826446",
        "5273114895302717",
        "5430972152222336",
        "5536426859893306",
    ]
    amex = [
        "371025944923273",
        "374745112042294",
        "340984902710890",
        "375767928645325",
        "370720852891659",
    ]
    missing = [np.nan]
    invalid = ["354 435"]

    creditcard = pd.Series(
        visa + mastercard + amex + missing + invalid, name="creditcard"
    )

    ### continuous
    cts = pd.Series([123.32, 23.243, 324.342, np.nan], name="cts")

    ### datetime
    datetime = pd.Series(
        [
            "3/11/2000",
            "3/12/2000",
            "3/13/2000",
            "",
            None,
            np.nan,
            "April/13/2011",
            "April/15/11",
        ],
        name="datetime",
    )

    ### Phone Number
    phonenumber = pd.Series(
        ["2068866666", "6508866666", "2068866666", "", np.NaN, np.nan, None],
        name="phone",
    )

    ### Lat Long
    latlong = pd.Series(
        [
            "69.196241,-125.017615",
            "5.2272595,-143.465712",
            "-33.9855425,-153.445155",
            "43.340610,86.460554",
            "24.2811855,-162.380403",
            "2.7849025,-7.328156",
            "45.033805,157.490179",
            "-1.818319,-80.681214",
            "-44.510428,-169.269477",
            "-56.3344375,-166.407038",
            "",
            np.NaN,
            None,
        ],
        name="latlon",
    )
    ### zip code
    zipcode = pd.Series([94065, 90210, np.NaN, None], name="zipcode")

    ### boolean
    boolean = pd.Series([True, False, True, False, np.NaN, None], name="bool")

    ### string
    string = pd.Series(
        [
            "S",
            "C",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "S",
            "C",
            "S",
            "S",
            "S",
            "S",
            "S",
            "S",
            "Q",
            "S",
            "S",
            "",
            np.NaN,
            None,
        ],
        name="string",
    )

    ### Address
    address = pd.Series(
        [
            "1 Miller Drive, New York, NY 12345",
            "1 Berkeley Street, Boston, MA 67891",
            "54305 Oxford Street, Seattle, WA 95132",
            "",
        ],
        name="address",
    )

    ### constant
    constant = pd.Series([1, 1, 1, 1, 1], name="constant")

    ### discrete
    discrete_numbers = pd.Series([35, 25, 13, 42], name="discrete")

    ### gis
    gis = pd.Series(
        [
            "69.196241,-125.017615",
            "5.2272595,-143.465712",
            "-33.9855425,-153.445155",
            "43.340610,86.460554",
            "24.2811855,-162.380403",
            "2.7849025,-7.328156",
            "45.033805,157.490179",
            "-1.818319,-80.681214",
            "-44.510428,-169.269477",
            "-56.3344375,-166.407038",
            "",
            np.NaN,
            None,
        ],
        name="gis",
    )

    ### ipaddress
    ip_address = pd.Series(
        ["2002:db8::", "192.168.0.1", "2001:db8::", "2002:db8::", np.NaN, None],
        name="ip_address",
    )

    ### ipaddressv4
    ip_address_v4 = pd.Series(
        ["192.168.0.1", "192.168.0.2", "192.168.0.3", "192.168.0.4", np.NaN, None],
        name="ip_address_v4",
    )

    ### ipaddressv6
    ip_address_v6 = pd.Series(
        ["2002:db8::", "2001:db8::", "2001:db8::", "2002:db8::", np.NaN, None],
        name="ip_address_v6",
    )

    ### DataFrame
    df = load_iris(as_frame=True).data

    ### schema
    schema = Schema.from_dict(
        {
            "Schema": [
                {
                    "dtype": "int64",
                    "feature_type": "Integer",
                    "name": "Age",
                    "domain": {"values": "Integer", "stats": None, "constraints": []},
                    "required": False,
                    "order": 3,
                    "description": "Age",
                },
                {
                    "dtype": "object",
                    "feature_type": "String",
                    "name": "Attrition",
                    "domain": {
                        "values": "String",
                        "stats": {"count": 1470, "unique": 2},
                        "constraints": [],
                    },
                    "required": False,
                    "order": 12,
                    "description": "Attrition",
                },
            ]
        }
    )

    schema_no_order = Schema.from_dict(
        {
            "Schema": [
                {
                    "dtype": "int64",
                    "feature_type": "Integer",
                    "name": "Age",
                    "domain": {"values": "Integer", "stats": None, "constraints": []},
                    "required": False,
                    "description": "Age",
                },
                {
                    "dtype": "object",
                    "feature_type": "String",
                    "name": "Attrition",
                    "domain": {
                        "values": "String",
                        "stats": {"count": 1470, "unique": 2},
                        "constraints": [],
                    },
                    "required": False,
                    "description": "Attrition",
                },
            ]
        }
    )

    ### invalid schema
    invalid_schema = Schema.from_dict(
        {
            "Schema": [
                {
                    "dtype": None,
                    "feature_type": None,
                    "name": "Age",
                    "domain": {"values": "Integer", "stats": None, "constraints": []},
                    "required": False,
                    "order": 1,
                    "description": "Age",
                }
            ]
        }
    )

    ## iris data
    X, y = load_iris(return_X_y=True)

    def setup_class(cls):
        cls.curdir = os.path.dirname(os.path.abspath(__file__))

    def setup_method(self):
        self.dir = tempfile.mkdtemp(prefix="data_schema")
        self.model_dir = tempfile.mkdtemp(prefix="model")
        self.model_artifact = get_model_artifact_instance(self.model_dir)

    def test_credit_card(self):
        self.creditcard.ads.feature_type = ["credit_card"]
        domain = self.creditcard.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "CreditCard"
        assert domain.stats["missing"] == 1
        assert domain.stats["count_Visa"] == 5
        assert domain.stats["count_Amex"] == 5
        assert domain.stats["count_MasterCard"] == 3
        assert domain.stats["count_Diners Club"] == 2
        assert domain.stats["count_Unknown"] == 1

    def test_continuous(self):
        self.cts.ads.feature_type = ["continuous"]
        domain = self.cts.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "Continuous"

    def test_datetime(self):
        self.datetime.ads.feature_type = ["date_time"]
        domain = self.datetime.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "DateTime"

    def test_phone_number(self):
        self.phonenumber.ads.feature_type = ["phone_number", "category"]
        domain = self.phonenumber.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "PhoneNumber"

    def test_lat_long(self):
        self.latlong.ads.feature_type = ["lat_long"]
        domain = self.latlong.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "LatLong"

    def test_zipcode(self):
        self.zipcode.ads.feature_type = ["zip_code"]
        domain = self.zipcode.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "ZipCode"

    @pytest.mark.skipif("NoDependency" in os.environ, reason="skip for dependency test")
    @pytest.mark.parametrize("feature_type", ["boolean", "category", "ordinal"])
    def test_categorical_bool_ordinal(self, feature_type):
        self.boolean.ads.feature_type = [feature_type]
        domain = self.boolean.ads.feature_domain()
        assert domain.constraints[0].expression == f"$x in [True, False]"
        assert domain.constraints[0].evaluate(x=True)
        assert domain.constraints[0].evaluate(x=False)

    def test_string(self):
        self.string.ads.feature_type = ["string"]
        domain = self.string.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "String"

    def test_address(self):
        self.address.ads.feature_type = ["address"]
        domain = self.address.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "Address"

    def test_constant(self):
        self.constant.ads.feature_type = ["constant"]
        domain = self.constant.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "Constant"

    def test_discrete(self):
        self.discrete_numbers.ads.feature_type = ["discrete"]
        domain = self.discrete_numbers.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "Discrete"

    def test_gis(self):
        self.gis.ads.feature_type = ["gis"]
        domain = self.gis.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "GIS"

    def test_ipaddress(self):
        self.ip_address.ads.feature_type = ["ip_address"]
        domain = self.ip_address.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "IpAddress"

    def test_ipaddress_v4(self):
        self.ip_address_v4.ads.feature_type = ["ip_address_v4"]
        domain = self.ip_address_v4.ads.feature_domain()
        assert domain.constraints == []
        assert domain.values == "IpAddressV4"

    def test_ipaddress_v6(self):
        self.ip_address_v6.ads.feature_type = ["ip_address_v6"]
        domain = self.ip_address_v6.ads.feature_domain()
        assert domain.to_dict() == {
            "values": "IpAddressV6",
            "stats": {"count": 6, "unique": 2, "missing": 2},
            "constraints": [],
        }

    def test_dataframe_schema(self):
        schema = self.df.ads.model_schema()
        assert set(self.df.columns).issubset(
            set([item["name"] for item in schema.to_dict()["schema"]])
        )

    def test_schema_to_dict(self):
        schema_dict = self.df.ads.model_schema().to_dict()
        assert isinstance(schema_dict, dict)

    def test_schema_loader(self):
        test_files_dir = os.path.join(self.curdir, "test_files")
        test_files = [
            f"{test_files_dir}/without_version_field.yaml",
            f"{test_files_dir}/with_version_field.yaml",
            f"{test_files_dir}/without_version_field.json",
            f"{test_files_dir}/with_version_field.json",
        ]

        for file in test_files:
            basename = os.path.basename(file)
            if os.path.splitext(file)[-1].lower() in [".json"]:
                schema_loader = JsonSchemaLoader()
            else:
                schema_loader = YamlSchemaLoader()

            schema_dict = schema_loader.load_schema(file)
            assert "version" in schema_dict.keys()
            if basename.startswith("without"):
                assert schema_dict["version"] == DEFAULT_SCHEMA_VERSION
            else:
                assert schema_dict["version"] == SCHEMA_VERSION

    def test_schema_iteration_by_order(self):
        schema = self.df.ads.model_schema()
        expected_order = 0
        for attr in schema:
            assert attr.order == expected_order
            expected_order += 1

    def test_schema_iteration_no_order(self):
        for attr in self.schema_no_order:
            assert isinstance(attr, Attribute)

    def test_schema_from_dict(self):
        assert "Attrition" in self.schema.keys
        assert "Age" in self.schema.keys
        with pytest.raises(ValueError):
            self.schema.add(self.schema["Attrition"])
        self.schema.add(self.schema["Attrition"], replace=True)
        with pytest.raises(TypeError):
            self.schema.add(self.schema["Attrition"].to_dict())

    def test_schema_json_file(self):
        schema_path = os.path.join(self.dir, "test.json")
        self.schema.to_json_file(schema_path)
        assert os.path.exists(schema_path)

        data_schema = Schema.from_file(schema_path)
        assert data_schema == self.schema

    def test_schema_yaml_file(self):
        schema_path = os.path.join(self.dir, "test.yaml")
        self.schema.to_yaml_file(schema_path)
        assert os.path.exists(schema_path)

        data_schema = Schema.from_file(schema_path)
        assert data_schema == self.schema

    def test_schema_yml_file(self):
        schema_path = os.path.join(self.dir, "test.yml")
        self.schema.to_yaml_file(schema_path)
        assert os.path.exists(schema_path)

        data_schema = Schema.from_file(schema_path)
        assert data_schema == self.schema

    def test_schema_invalid_format(self):
        schema_path = os.path.join(self.dir, "invalid_path.yaml")
        with pytest.raises(FileNotFoundError):
            Schema.from_file(schema_path)
        schema_path = os.path.join(self.dir, "invalid_path.json")
        with pytest.raises(FileNotFoundError):
            Schema.from_file(schema_path)

    def test_schema_invalid(self):
        schema_path = os.path.join(self.dir, "invalid_schema.json")
        self.invalid_schema.to_json_file(schema_path)
        with pytest.raises(ValueError):
            Schema.from_file(schema_path)

    def test_populate_schema_numpy(self):
        self.model_artifact.populate_schema(X_sample=self.X, y_sample=self.y)
        assert isinstance(self.model_artifact.schema_input, Schema)
        assert isinstance(self.model_artifact.schema_output, Schema)

        # Test wide data
        with patch.object(Schema, "validate_size", side_effect=SchemaSizeTooLarge(100)):
            with patch.object(logger, "warning") as mock_warning:
                self.model_artifact.populate_schema(X_sample=self.X, y_sample=self.y)
                mock_warning.assert_called()

    def test_populate_schema_adsdata(self):
        data = ADSData(self.X, self.y)
        self.model_artifact.populate_schema(data_sample=data)
        assert isinstance(self.model_artifact.schema_input, Schema)
        assert isinstance(self.model_artifact.schema_output, Schema)

        # Test wide data
        with patch.object(Schema, "validate_size", side_effect=SchemaSizeTooLarge(100)):
            with patch.object(logger, "warning") as mock_warning:
                self.model_artifact.populate_schema(data_sample=data)
                mock_warning.assert_called()

    def test_populate_schema_list(self):
        self.model_artifact.populate_schema(X_sample=list(self.X))
        assert isinstance(self.model_artifact.schema_input, Schema)
        assert self.model_artifact.schema_output == Schema()

        # Test wide data
        with patch.object(Schema, "validate_size", side_effect=SchemaSizeTooLarge(100)):
            with patch.object(logger, "warning") as mock_warning:
                self.model_artifact.populate_schema(X_sample=list(self.X))
                mock_warning.assert_called()

    @pytest.mark.skipif("NoDependency" in os.environ, reason="skip for dependency test")
    def test_populate_schema_dask_tuple(self):
        import dask.dataframe as dd

        self.model_artifact.populate_schema(
            X_sample=dd.from_array(self.X), y_sample=tuple(self.y)
        )
        assert isinstance(self.model_artifact.schema_input, Schema)
        assert isinstance(self.model_artifact.schema_output, Schema)

        # Test wide data
        with patch.object(Schema, "validate_size", side_effect=SchemaSizeTooLarge(100)):
            with patch.object(logger, "warning") as mock_warning:
                self.model_artifact.populate_schema(
                    X_sample=dd.from_array(self.X), y_sample=tuple(self.y)
                )
                mock_warning.assert_called()

    def test_simple_constraint(self):
        self.df["sepal length (cm)"].ads.feature_type = ["category"]
        domain = self.df["sepal length (cm)"].ads.feature_domain()
        assert isinstance(domain.constraints[0], Expression)

    @pytest.mark.skipif("NoDependency" in os.environ, reason="skip for dependency test")
    def test_expression(self):
        domain = self.df["sepal length (cm)"].ads.feature_domain()
        domain.constraints.append(Expression("$x < 8"))
        assert domain.constraints[1].evaluate(x="9") is False
        assert domain.constraints[1].evaluate(x="7") is True

    @pytest.mark.skipif("NoDependency" in os.environ, reason="skip for dependency test")
    def test_from_dict(self):
        self.df["sepal length (cm)"].ads.feature_type = ["category"]
        schema = self.df.ads.model_schema()
        new_schema = Schema.from_dict(schema.to_dict())

        assert isinstance(
            new_schema["sepal length (cm)"].domain.constraints[0], Expression
        )
        assert new_schema["sepal length (cm)"].domain.constraints[0].evaluate(x="5.0")

    def teardown_method(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir, ignore_errors=True)
