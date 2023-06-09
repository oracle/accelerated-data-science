#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.oracledb.oracle_db import OracleRDBMSConnection
from tests.integration.config import secrets
from base64 import b64encode
from datetime import datetime
import logging
import os
import pandas as pd
import pytest
import uuid


def get_test_dataset_path(file_name):
    return os.path.join(
        f"{os.path.dirname(os.path.abspath(__file__))}/../vor_datasets/", file_name
    )


@pytest.mark.thickclient
def test_thick_client_enabled(caplog):
    with caplog.at_level(logging.DEBUG, logger="ads.oracle_connector"):
        pd.DataFrame.ads.read_sql(
            "SELECT * FROM SH.SALES",
            connection_parameters={
                "user_name": secrets.other.username,
                "password": secrets.other.password or os.environ["password"],
                "service_name": secrets.other.service_name,
                "wallet_location": secrets.other.wallet_file,
            },
        )
    assert (
        "ads.oracle_connector",
        logging.INFO,
        "Running oracledb driver in thick mode. For mTLS based connection, thick mode is default.",
    ) in caplog.record_tuples


@pytest.mark.thickclient
class TestOraclePandasAccess:
    quick_test_files = ["vor_iris.csv"]

    select_test_file = [
        (
            "vor_iris.csv",
            ("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            ("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            5,
            150,
        ),
        ("vor_iris.csv", ("VARIETY",), ("VARIETY",), 1, 150),
        (
            "vor_iris.csv",
            ("SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            ("SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            4,
            150,
        ),
        (
            "vor_iris.csv",
            ("*"),
            ("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            5,
            150,
        ),
    ]

    select_test_file_bind = [
        (
            "vor_iris.csv",
            "SELECT * from $TABLENAME where sepal_length > :slength and petal_length > :plength",
            {"slength": 1.5, "plength": 2.5},
            100,
            5,
        ),
        (
            "vor_breast_cancer.csv",
            "SELECT * from $TABLENAME where ID = :ID",
            {"ID": "84358402"},
            1,
            32,
        ),
    ]

    select_test_file_chunksize_1 = [
        (
            "vor_iris.csv",
            ("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            10,
            15,
            150,
        ),
    ]
    select_test_file_chunksize = [
        (
            "vor_iris.csv",
            ("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            10,
            15,
            150,
        ),
        (
            "vor_iris.csv",
            ("SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            9,
            17,
            150,
        ),
        (
            "vor_iris.csv",
            ("SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            151,
            1,
            150,
        ),
        (
            "vor_iris.csv",
            ("SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH", "VARIETY"),
            150,
            1,
            150,
        ),
    ]

    failure_dataset = [
        (
            pd.DataFrame(
                {
                    "c1": [1, 2],
                    "c2": ["a", "b"],
                    "c3": [3.14, 2.78],
                    "c4": ["hello", "world"],
                    "c5": ["red", "blue"],
                    "c6": [True, False],
                    "c7": [datetime(1991, 5, 6, 1, 1), datetime(1965, 9, 19, 0, 0)],
                }
            ),
            "fail",
            [
                (1, "a", 3.14, "hello", "red", 1, datetime(1991, 5, 6, 1, 1)),
                (2, "b", 2.78, "world", "blue", 0, datetime(1965, 9, 19, 0, 0)),
            ],
        ),
    ]
    append_dataset = [
        (
            pd.DataFrame(
                {
                    "c1": [1, 2],
                    "c2": ["a", "b"],
                    "c3": [3.14, 2.78],
                    "c4": ["hello", "world"],
                    "c5": ["red", "blue"],
                    "c6": [True, False],
                    "c7": [datetime(1991, 5, 6, 1, 1), datetime(1965, 9, 19, 0, 0)],
                }
            ),
            "append",
            [
                (1, "a", 3.14, "hello", "red", 1, datetime(1991, 5, 6, 1, 1)),
                (2, "b", 2.78, "world", "blue", 0, datetime(1965, 9, 19, 0, 0)),
            ],
        ),
    ]

    dataset = [
        (
            pd.DataFrame(
                {
                    "c1": [1, 2],
                    "c2": ["a", "b"],
                    "c3": [3.14, 2.78],
                    "c4": ["hello", "world"],
                    "c5": ["red", "blue"],
                    "c6": [True, False],
                    "c7": [datetime(1991, 5, 6, 1, 1), datetime(1965, 9, 19, 0, 0)],
                }
            ),
            "append",
            [
                (1, "a", 3.14, "hello", "red", 1, datetime(1991, 5, 6, 1, 1)),
                (2, "b", 2.78, "world", "blue", 0, datetime(1965, 9, 19, 0, 0)),
            ],
        ),
        (
            pd.DataFrame(
                {
                    "c1": [1, 2],
                    "c2": ["a", "b"],
                    "c3": [3.14, 2.78],
                    "c4": ["hello", "world"],
                    "c5": ["red", "blue"],
                    "c6": [True, False],
                    "c7": [datetime(1991, 5, 6, 1, 1), datetime(1965, 9, 19, 0, 0)],
                }
            ),
            "replace",
            [
                (1, "a", 3.14, "hello", "red", 1, datetime(1991, 5, 6, 1, 1)),
                (2, "b", 2.78, "world", "blue", 0, datetime(1965, 9, 19, 0, 0)),
            ],
        ),
    ]

    complexdataset = [
        (
            pd.DataFrame(
                {
                    "c1": [1, 2],
                    "c2": [["a", "b"], ["x", "y"]],
                    "c3": [{"foo": "12"}, {"bar": 42}],
                }
            ),
            "replace",
            [
                (1, str(["a", "b"]), str({"foo": "12"})),
                (2, str(["x", "y"]), str({"bar": 42})),
            ],
        ),
    ]

    @pytest.fixture
    def database_connection(self):
        return OracleRDBMSConnection(
            user_name=secrets.other.username,
            password=secrets.other.password or os.environ["password"],
            service_name=secrets.other.service_name,
            wallet_file=secrets.other.wallet_file,
        )

    @pytest.fixture
    def connection_parameters(self):
        return {
            "user_name": secrets.other.username,
            "password": secrets.other.password or os.environ["password"],
            "service_name": secrets.other.service_name,
            "wallet_location": secrets.other.wallet_file,
        }

    @pytest.fixture(params=quick_test_files)
    def quick_test_files(self, request):
        return request.param

    @pytest.fixture(params=select_test_file)
    def select_test_file(self, request):
        return request.param

    @pytest.fixture(params=select_test_file_chunksize)
    def select_test_file_chunksize(self, request):
        return request.param

    @pytest.fixture(params=select_test_file_bind)
    def select_test_file_bind(self, request):
        return request.param

    @pytest.fixture(params=failure_dataset)
    def failure_dataset(self, request):
        return request.param

    @pytest.fixture(params=append_dataset)
    def append_dataset(self, request):
        return request.param

    @pytest.fixture(params=dataset)
    def dataset(self, request):
        return request.param

    @pytest.fixture(params=complexdataset)
    def complexdataset(self, request):
        return request.param

    def uid(self):
        return str(uuid.uuid4()).split("-")[-1]

    def test_select_pandas_accessor(
        self, select_test_file, database_connection, connection_parameters
    ):
        data = pd.read_csv(get_test_dataset_path(select_test_file[0]))
        table_name = f"ADSTEST_READ_{os.path.splitext(os.path.basename(select_test_file[0]))[0]}_{self.uid()}"
        try:
            database_connection.insert(table_name, data, if_exists="replace")
        except:
            logging.info(f"Data Already exists")
        # database_connection.insert(table_name, data)
        output_data = pd.DataFrame.ads.read_sql(
            f"SELECT {','.join(select_test_file[1])} from {table_name}",
            bind_variables={},
            connection_parameters=connection_parameters,
        )
        assert len(output_data) == select_test_file[4], "The number of rows dont match"
        assert (
            len(output_data.columns) == select_test_file[3]
        ), "Mismatch in the number of columns in pandas dataframe"
        assert tuple(select_test_file[2]) == tuple(
            [str(col) for col in output_data.columns]
        ), "Pandas column names dont match with the table column name"
        cursor = database_connection.cursor()

        cursor.execute(f"DROP Table {table_name}")

    def test_insert_pandas_accessor(
        self, quick_test_files, database_connection, connection_parameters
    ):
        data = pd.read_csv(get_test_dataset_path(quick_test_files))
        table_name = f"ADSTEST_PANDAS_INSERT_{os.path.splitext(os.path.basename(quick_test_files))[0]}_{self.uid()}"
        data.ads.to_sql(
            table_name, connection_parameters=connection_parameters, if_exists="replace"
        )

        cursor = database_connection.cursor()
        assert cursor.execute(f"SELECT COUNT(1) from {table_name}").fetchall()[0][
            0
        ] == len(data)

        cursor.execute(f"DROP Table {table_name}")

    def test_insert_pandas_accessor_batch_size(
        self, quick_test_files, database_connection, connection_parameters
    ):
        data = pd.read_csv(get_test_dataset_path(quick_test_files))
        table_name = f"ADSTEST_PANDAS_INSERT_{os.path.splitext(os.path.basename(quick_test_files))[0]}_{self.uid()}"
        data.ads.to_sql(
            table_name,
            connection_parameters=connection_parameters,
            if_exists="replace",
            batch_size=50,
        )

        cursor = database_connection.cursor()
        assert cursor.execute(f"SELECT COUNT(1) from {table_name}").fetchall()[0][
            0
        ] == len(data)

        cursor.execute(f"DROP Table {table_name}")

    def test_pandas_test_if_exists_fail(
        self, failure_dataset, database_connection, connection_parameters
    ):
        table_name = f"ADSTEST_PANDAS_TYPE_CHECK_{self.uid()}"
        dataset = failure_dataset
        # Checking the case where the table already exists. In such case,
        # `if_exists==fail` should throw an error
        database_connection.insert(table_name, dataset[0], if_exists="replace")
        with pytest.raises(
            ValueError,
            match=f"Table {table_name} already exists. Set `if_exists`='replace' or 'append' to replace or append to the existing table",
        ):
            dataset[0].ads.to_sql(
                table_name,
                connection_parameters=connection_parameters,
                if_exists="fail",
            )
        cursor = database_connection.cursor()
        cursor.execute(f"DROP Table {table_name}")

    def test_pandas_test_if_exists_invalid(
        self, failure_dataset, database_connection, connection_parameters
    ):
        table_name = f"ADSTEST_PANDAS_TYPE_CHECK_{self.uid()}"
        dataset = failure_dataset
        with pytest.raises(
            ValueError,
            match=f"Unknown option `if_exists`=xxfail. Valid options are 'fail', 'replace', 'append'",
        ):
            dataset[0].ads.to_sql(
                table_name,
                connection_parameters=connection_parameters,
                if_exists="xxfail",
            )

    def test_pandas_test_if_exists_append(
        self, append_dataset, database_connection, connection_parameters
    ):
        dataset = append_dataset
        table_name = f"ADSTEST_PANDAS_TYPE_CHECK_{self.uid()}"
        dataset[0].ads.to_sql(
            table_name,
            connection_parameters=connection_parameters,
            if_exists=dataset[1],
        )
        dataset[0].ads.to_sql(
            table_name,
            connection_parameters=connection_parameters,
            if_exists=dataset[1],
        )
        cursor = database_connection.cursor()
        assert cursor.execute(f"SELECT COUNT(1) from {table_name}").fetchall()[0][
            0
        ] == 2 * len(dataset[0])
        cursor.execute(f"DROP Table {table_name}")

    def test_pandas_to_db_type_conversion(
        self, dataset, database_connection, connection_parameters
    ):
        data = dataset[0]
        expected_data = dataset[2]
        data = data.astype(dtype={"c5": "category"})
        table_name = f"ADSTEST_PANDAS_TYPE_CHECK_{self.uid()}"
        data.ads.to_sql(
            table_name,
            connection_parameters=connection_parameters,
            if_exists=dataset[1],
        )
        cursor = database_connection.cursor()
        assert cursor.execute(f"SELECT COUNT(1) from {table_name}").fetchall()[0][
            0
        ] == len(data)

        cursor.execute(
            f"SELECT  {','.join([str(col) for col in data.columns])} from {table_name} order by c1 asc"
        )
        i = 0
        for row in cursor:
            assert row == expected_data[i]
            i += 1
        cursor.execute(f"DROP Table {table_name}")

    @pytest.mark.skip(
        reason="Need to handle object type. But sometimes blindly converting object col to str produces 'nan' values. nan should be null in the database"
    )
    def test_pandas_to_db_complex(
        self, complexdataset, database_connection, connection_parameters
    ):
        dataset = complexdataset
        data = dataset[0]
        expected_data = dataset[2]
        table_name = f"ADSTEST_PANDAS_TYPE_CHECK_{self.uid()}"
        data.ads.to_sql(
            table_name,
            connection_parameters=connection_parameters,
            if_exists=dataset[1],
        )
        cursor = database_connection.cursor()
        assert cursor.execute(f"SELECT COUNT(1) from {table_name}").fetchall()[0][
            0
        ] == len(data)

        cursor.execute(
            f"SELECT  {','.join([str(col) for col in data.columns])} from {table_name} order by c1 asc"
        )
        i = 0
        for row in cursor:
            assert row == expected_data[i]
            i += 1
        cursor.execute(f"DROP Table {table_name}")

    def test_insert_long_string_dataset(self, connection_parameters):
        """Maximum size for VARCHAR2 in Oracle DB is 4000 bytes.
        On most platforms, the maximum length of a VARCHAR2 value is 65535 bytes.
        This test validates expected error message, when string too long for its datatype.
        """
        dataset = pd.DataFrame(
            {
                "c1": [1],
                "c2": [b64encode(os.urandom(65536)).decode("utf-8")],
            }
        )

        with pytest.raises(Exception) as excinfo:
            dataset.ads.to_sql(
                table_name="ADSTEST_LONG_STRING_CHECK",
                connection_parameters=connection_parameters,
                if_exists="replace",
            )
        assert "specified length too long for its datatype" in str(excinfo.value)

    @classmethod
    def teardown_class(cls):
        connection = OracleRDBMSConnection(
            user_name=secrets.other.username,
            password=secrets.other.password or os.environ["password"],
            service_name=secrets.other.service_name,
            wallet_file=secrets.other.wallet_file,
        )

        cursor = connection.cursor()
        for table in [
            # "ADSTEST_VOR_FLIGHTS5K",
            "ADSTEST_VOR_BREAST_CANCER",
            "ADSTEST_VOR_IRIS",
            "ADSTEST_VOR_EMPLOYEE_ATTRITION",
            "ADSTEST_READ_VOR_IRIS",
            "ADSTEST_READ_VOR_EMPLOYEE_ATTRITION",
            "ADSTEST_READ_VOR_BREAST_CANCER",
            "ADSTEST_PANDAS_TYPE_CHECK",
        ]:
            try:
                cursor.execute(f"DROP Table {table}")
            except:
                pass
