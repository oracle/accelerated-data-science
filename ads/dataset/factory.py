#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import os
import re
import warnings
import oci
import datetime
import pandas as pd
from fsspec.utils import infer_storage_options
import inspect
import fsspec

from ads.common import utils
from ads.common.utils import is_same_class
from ads.dataset import logger
from ads.dataset.classification_dataset import (
    BinaryClassificationDataset,
    MultiClassClassificationDataset,
    BinaryTextClassificationDataset,
    MultiClassTextClassificationDataset,
)
from ads.dataset.dataset import ADSDataset
from ads.dataset.forecasting_dataset import ForecastingDataset
from ads.dataset.helper import (
    get_feature_type,
    is_text_data,
    generate_sample,
    DatasetDefaults,
    ElaboratedPath,
    DatasetLoadException,
)
from ads.dataset.regression_dataset import RegressionDataset
from ads.type_discovery.type_discovery_driver import TypeDiscoveryDriver
from ads.type_discovery.typed_feature import (
    ContinuousTypedFeature,
    DateTimeTypedFeature,
    CategoricalTypedFeature,
    OrdinalTypedFeature,
    GISTypedFeature,
    DocumentTypedFeature,
)
from ads.type_discovery.typed_feature import TypedFeature
from typing import Callable, Tuple
from ocifs import OCIFileSystem
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.decorator.deprecate import deprecated

default_snapshots_dir = None
default_storage_options = None
mindate = datetime.date(datetime.MINYEAR, 1, 1)


warnings.warn(
    (
        "The `ads.dataset.factory` is deprecated in `oracle-ads 2.8.8` and will be removed in `oracle-ads 3.0`."
        "Use Pandas to read from local files or object storage directly. "
        "Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/loading_data/connect.html."
    ),
    DeprecationWarning,
    stacklevel=2,
)


class DatasetFactory:
    @staticmethod
    @deprecated(
        "2.6.6",
        details="Deprecated in favor of using Pandas. Pandas supports reading from object storage directly. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/loading_data/connect.html",
    )
    def open(
        source,
        target=None,
        format="infer",
        reader_fn: Callable = None,
        name: str = None,
        description="",
        npartitions: int = None,
        type_discovery=True,
        html_table_index=None,
        column_names="infer",
        sample_max_rows=10000,
        positive_class=None,
        transformer_pipeline=None,
        types={},
        **kwargs,
    ):
        """
        Returns an object of ADSDataset or ADSDatasetWithTarget  read from the given path

        .. deprecated:: 2.6.6
            "Deprecated in favor of using Pandas. Pandas supports reading from object storage directly.
            Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/loading_data/connect.html",

        Parameters
        ----------
        source: Union[str, pandas.DataFrame, h2o.DataFrame, pyspark.sql.dataframe.DataFrame]
            If str, URI for the dataset. The dataset could be read from local or network file system, hdfs, s3, gcs and optionally pyspark in pyspark
            conda env
        target: str, optional
            Name of the target in dataset.
            If set an ADSDatasetWithTarget object is returned, otherwise an ADSDataset object is returned which can be
            used to understand the dataset through visualizations
        format: str, default: infer
            Format of the dataset.
            Supported formats: CSV, TSV, Parquet, libsvm, JSON, XLS/XLSX (Excel), HDF5, SQL, XML,
            Apache server log files (clf, log), ARFF.
            By default, the format would be inferred from the ending of the dataset file path.
        reader_fn: Callable, default: None
            The user may pass in their own custom reader function.
            It must accept `(path, **kwarg)` and return a pandas DataFrame
        name: str, optional default: ""
        description: str, optional default: ""
            Text describing the dataset
        npartitions: int, deprecated
            Number of partitions to split the data
            By default this is set to the max number of cores supported by the backend compute accelerator
        type_discovery: bool, default: True
            If false, the data types of the dataframe are used as such.
            By default, the dataframe columns are associated with the best suited data types. Associating the features
            with the disovered datatypes would impact visualizations and model prediction.
        html_table_index: int, optional
            The index of the dataframe table in html content. This is used when the format of dataset is html
        column_names: 'infer', list of str or None, default: 'infer'
            Supported only for CSV and TSV.
            List of column names to use.
            By default, column names are inferred from the first line of the file.
            If set to None, column names would be auto-generated instead of inferring from file.
            If the file already contains a column header, specify header=0 to ignore the existing column names.
        sample_max_rows: int, default: 10000, use -1 auto calculate sample size, use 0 (zero) for no sampling
            Sample size of the dataframe to use for visualization and optimization.
        positive_class: Any, optional
            Label in target for binary classification problems which should be identified as positive for modeling.
            By default, the first unique value is considered as the positive label.
        types: dict, optional
            Dictionary of <feature_name> : <data_type> to override the data type of features.
        transformer_pipeline: datasets.pipeline.TransformerPipeline, optional
            A pipeline of transformations done outside the sdk and need to be applied at the time of scoring
        storage_options: dict, default: varies by source type
            Parameters passed on to the backend filesystem class.
        sep: str
            Delimiting character for parsing the input file.
        kwargs: additional keyword arguments that would be passed to underlying dataframe read API
            based on the format of the dataset

        Returns
        -------
        dataset : An instance of ADSDataset
        (or)
        dataset_with_target : An instance of ADSDatasetWithTarget

        Examples
        --------
        >>> ds = DatasetFactory.open("/path/to/data.data", format='csv', delimiter=" ",
        ...          na_values="n/a", skipinitialspace=True)

        >>> ds = DatasetFactory.open("/path/to/data.csv", target="col_1", prefix="col_",
        ...           skiprows=1, encoding="ISO-8859-1")

        >>> ds = DatasetFactory.open("oci://bucket@namespace/path/to/data.tsv",
        ...         column_names=["col1", "col2", "col3"], header=0)

        >>> ds = DatasetFactory.open("oci://bucket@namespace/path/to/data.csv",
        ...         storage_options={"config": "~/.oci/config",
        ...         "profile": "USER_2"}, delimiter = ';')

        >>> ds = DatasetFactory.open("/path/to/data.parquet", engine='pyarrow',
        ...         types={"col1": "ordinal",
        ...                "col2": "categorical",
        ...                "col3" : "continuous",
        ...                "col4" : "float64"})

        >>> ds = DatasetFactory.open(df, target="class", sample_max_rows=5000,
        ...          positive_class="yes")

        >>> ds = DatasetFactory.open("s3://path/to/data.json.gz", format="json",
        ...         compression="gzip", orient="records")
        """
        if npartitions:
            warnings.warn(
                "Variable `npartitions` is deprecated and will not be used",
                DeprecationWarning,
                stacklevel=2,
            )
        if (
            "storage_options" not in kwargs
            and type(source) is str
            and len(source) > 6
            and source[:6] == "oci://"
        ):
            kwargs["storage_options"] = {"config": {}}

        if isinstance(source, str) or isinstance(source, list):
            progress = utils.get_progress_bar(4)
            progress.update("Opening data")
            path = ElaboratedPath(source, format=format, **kwargs)
            reader_fn = (
                get_format_reader(path=path, **kwargs)
                if reader_fn is None
                else reader_fn
            )
            df = load_dataset(path=path, reader_fn=reader_fn, **kwargs)
            name = path.name
        elif isinstance(source, pd.DataFrame):
            progress = utils.get_progress_bar(4)
            progress.update("Partitioning data")
            df = source
            name = "User Provided DataFrame" if name is None else name
        else:
            raise TypeError(
                f"The Source type: {type(source)} is not supported for DatasetFactory."
            )
        shape = df.shape
        return DatasetFactory._build_dataset(
            df=df,
            shape=shape,
            target=target,
            sample_max_rows=sample_max_rows,
            type_discovery=type_discovery,
            types=types,
            positive_class=positive_class,
            name=name,
            transformer_pipeline=transformer_pipeline,
            description=description,
            progress=progress,
            **utils.inject_and_copy_kwargs(
                kwargs,
                **{"html_table_index": html_table_index, "column_names": column_names},
            ),
        )

    @staticmethod
    def open_to_pandas(
        source: str, format: str = None, reader_fn: Callable = None, **kwargs
    ) -> pd.DataFrame:
        path = ElaboratedPath(source, format=format, **kwargs)
        reader_fn = (
            get_format_reader(path=path, **kwargs) if reader_fn is None else reader_fn
        )
        df = load_dataset(path=path, reader_fn=reader_fn, **kwargs)
        return df

    @staticmethod
    def from_dataframe(df, target: str = None, **kwargs):
        """
        Returns an object of ADSDatasetWithTarget or ADSDataset given a pandas.DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
        target: str
        kwargs: dict
            See DatasetFactory.open() for supported kwargs

        Returns
        -------
        dataset: an object of ADSDataset target is not specified, otherwise an object of ADSDatasetWithTarget tagged
                  according to the type of target

        Examples
        --------
        >>> df = pd.DataFrame(data)
        >>> ds = from_dataframe(df)
        """
        return DatasetFactory.open(df, target=target, **kwargs)

    @staticmethod
    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    @runtime_dependency(
        module="ipywidgets",
        object="HTML",
        is_for_notebook_only=True,
        install_from=OptionalDependency.NOTEBOOK,
    )
    def list_snapshots(snapshot_dir=None, name="", storage_options=None, **kwargs):
        """
        Displays the URIs for dataset snapshots under the given directory path.

        Parameters
        ----------
        snapshot_dir: str
            Return all dataset snapshots created using ADSDataset.snapshot() within this directory.
            The path can contain protocols such as oci, s3.
        name: str, optional
            The list of snapshots in the directory gets filtered by the name. Accepts glob expressions.
            default = `"ads_"`
        storage_options: dict
            Parameters passed on to the backend filesystem class.

        Example
        --------
        >>> DatasetFactory.list_snapshots(snapshot_dir="oci://my_bucket/snapshots_dir",
        ...             name="ads_iris_")

        Returns a list of all snapshots (recursively) saved to obj storage bucket `"my_bucket"` with prefix
        `"/snapshots_dir/ads_iris_**"` sorted by time created.
        """
        if snapshot_dir is None:
            snapshot_dir = default_snapshots_dir
            if snapshot_dir is None:
                raise ValueError(
                    "Specify snapshot_dir or use DatasetFactory.set_default_storage() to set default \
                                      storage options"
                )
            else:
                logger.info("Using default snapshots dir %s" % snapshot_dir)
        if storage_options is None:
            if default_storage_options is not None:
                storage_options = default_storage_options
                logger.info("Using default storage options")
            else:
                storage_options = dict()
        assert isinstance(storage_options, dict), (
            "The storage options parameter must be a dictionary. You can set "
            "this gloabally by calling DatasetFactory.set_default_storage("
            "storage_options={'config': 'location'}). "
        )
        url_options = infer_storage_options(snapshot_dir)
        protocol = url_options.pop("protocol", None)

        fs = OCIFileSystem(config=storage_options.get("config", None))
        kwargs.update({"refresh": True})
        obj_list = [
            (k, v.get("timeCreated", mindate).strftime("%Y-%m-%d %H:%M:%S"))
            for k, v in fs.glob(
                os.path.join(snapshot_dir, name + "**"), detail=True, **kwargs
            ).items()
            if v["type"] == "file"
        ]

        files = []
        for file, file_time in obj_list:
            if protocol in ["oci"]:
                r1 = re.compile(r"/part\.[0-9]{1,6}\.parquet$")
                parquet_part = r1.search(file)
                if parquet_part is not None:
                    parquet_filename = file[: parquet_part.start()]
                elif file.endswith("/_common_metadata"):
                    parquet_filename = file[: -len("/_common_metadata")]
                elif file.endswith("/_metadata"):
                    parquet_filename = file[: -len("/_metadata")]
                else:
                    parquet_filename = file
            else:
                parquet_filename = file
            parent_path = "%s://" % protocol
            files.append((parent_path + parquet_filename, file_time))
        files.sort(key=lambda x: x[1] or mindate, reverse=True)
        list_df = pd.DataFrame(files, columns=["Name", "Created Time"])
        list_df = list_df.drop_duplicates(subset=["Name"]).reset_index()
        if len(list_df) == 0:
            print(f"No snapshots found at: {os.path.join(snapshot_dir, name)}")

        # display in HTML format if sdk is run in notebook mode
        if utils.is_notebook():
            from IPython.core.display import display

            display(
                HTML(
                    list_df.style.set_table_attributes("class=table")
                    .hide_index()
                    .render()
                )
            )
        return list_df

    @staticmethod
    def download(remote_path, local_path, storage=None, overwrite=False):
        """
        Download a remote file or directory to local storage.

        Parameters
        ---------
        remote_path: str
            Supports protocols like oci, s3, also supports glob expressions
        local_path: str
            Supports glob expressions
        storage: dict
            Parameters passed on to the  backend remote filesystem class.
        overwrite: bool, default False
            If True, the method will overwrite any existing files in the local_path

        Examples
        ---------
        >>> DatasetFactory.download("oci://Bucket/prefix/to/data/*.csv",
        ...         "/home/datascience/data/")
        """
        if storage is None:
            if default_storage_options is not None:
                storage = default_storage_options
                logger.info("Using default storage options")
            else:
                storage = dict()

        remote_files = fsspec.open_files(
            remote_path, mode="rb", name_function=lambda i: "", **storage
        )
        if len(remote_files) < 1:
            raise FileNotFoundError(remote_path)
        display_error, error_msg = DatasetFactory._download_files(
            remote_files=remote_files, local_path=local_path, overwrite=overwrite
        )
        if display_error:
            logger.error(error_msg)
        else:
            logger.info(f"Download {remote_path} to {local_path}.")

    @staticmethod
    def _download_files(remote_files, local_path, overwrite=False):
        display_error, error_msg = False, ""
        for remote_file in remote_files:
            bucket_idx = remote_file.path.find("/")
            suffix = remote_file.path[bucket_idx + 1 :]
            try:
                with remote_file as f1:
                    local_filepath = (
                        os.path.join(local_path, suffix) if suffix else local_path
                    )
                    if os.path.exists(local_filepath) and not overwrite:
                        raise FileExistsError(
                            f"Trying to overwrite files in {local_filepath}. If you'd like to "
                            f"overwrite these files, set force_overwrite to True."
                        )
                    os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
                    with open(local_filepath, "wb") as f2:
                        f2.write(f1.read())
            except oci.exceptions.ServiceError as e:
                raise FileNotFoundError(f"Unable to open file: {remote_file.path}")
        return display_error, error_msg

    @staticmethod
    def upload(local_file_or_dir, remote_file_or_dir, storage_options=None):
        """
        Upload local file or directory to remote storage

        Parameters
        ---------
        local_file_or_dir: str
            Supports glob expressions
        remote_file_or_dir: str
            Supports protocols like oci, s3, also supports glob expressions
        storage_options: dict
            Parameters passed on to the  backend remote filesystem class.
        """
        if not os.path.exists(local_file_or_dir):
            raise ValueError("File/Directory does not exist: %s" % local_file_or_dir)
        if storage_options is None and default_storage_options is not None:
            storage_options = default_storage_options
            logger.info("Using default storage options")

        if os.path.isdir(local_file_or_dir):
            for subdir, dirs, files in os.walk(local_file_or_dir):
                for file in files:
                    if os.path.abspath(subdir) == os.path.abspath(local_file_or_dir):
                        path = file
                    else:
                        path = os.path.join(
                            os.path.abspath(subdir).split("/", 2)[2], file
                        )
                    DatasetFactory._upload_file(
                        os.path.join(subdir, file),
                        os.path.join(remote_file_or_dir, path),
                        storage_options=storage_options,
                    )
        else:
            DatasetFactory._upload_file(
                local_file_or_dir, remote_file_or_dir, storage_options=storage_options
            )

    @staticmethod
    def set_default_storage(snapshots_dir=None, storage_options=None):
        """
        Set default storage directory and options.

        Both snapshots_dir and storage_options can be overridden at the API scope.

        Parameters
        ----------
        snapshots_dir: str
            Path for the snapshots directory. Can contain protocols such as oci, s3
        storage_options: dict, optional
             Parameters passed on to the backend filesystem class.
        """
        global default_snapshots_dir
        default_snapshots_dir = snapshots_dir
        global default_storage_options
        if storage_options is not None:
            assert isinstance(storage_options, dict), (
                f"The storage options parameter must be a dictionary. Instead "
                f"we got the type: {type(storage_options)} "
            )
        default_storage_options = storage_options

    @classmethod
    def _upload_file(cls, local_file, remote_file, storage_options=None):
        kwargs = {}
        if storage_options is not None:
            kwargs = {"storage_options": storage_options}
        remote_file_handler = fsspec.open_files(
            remote_file + "*", mode="wb", name_function=lambda i: "", **kwargs
        )[0]
        with remote_file_handler as f1:
            with open(local_file, "rb") as f2:
                for line in f2:
                    f1.write(line)
        print("Uploaded %s to %s" % (local_file, remote_file))

    @classmethod
    def _build_dataset(
        cls,
        df: pd.DataFrame,
        shape: Tuple[int, int],
        target: str = None,
        progress=None,
        **kwargs,
    ):
        n = shape[0]
        if progress:
            progress.update("Generating data sample")

        sampled_df = generate_sample(
            df,
            n,
            DatasetDefaults.sampling_confidence_level,
            DatasetDefaults.sampling_confidence_interval,
            **kwargs,
        )

        if target is None:
            if progress:
                progress.update("Building the dataset with no target.")
            result = ADSDataset(df=df, sampled_df=sampled_df, shape=shape, **kwargs)
            if progress:
                progress.update("Done")
            logger.info(
                "Use `set_target()` to type the dataset for a particular learning task."
            )
            return result

        if progress:
            progress.update("Building dataset")

        discover_target_type = kwargs["type_discovery"]
        if target in kwargs["types"]:
            sampled_df[target] = sampled_df[target].astype(kwargs["types"][target])
            discover_target_type = False

        # if type discovery is turned off, infer type from pandas dtype
        target_type = DatasetFactory.infer_target_type(
            target, sampled_df[target], discover_target_type
        )

        result = DatasetFactory._get_dataset(
            df=df,
            sampled_df=sampled_df,
            target=target,
            target_type=target_type,
            shape=shape,
            **kwargs,
        )
        if progress:
            progress.update("Done")
        logger.info(
            "Use `suggest_recommendations()` to view and apply recommendations for dataset optimization."
        )
        return result

    @classmethod
    def infer_target_type(cls, target, target_series, discover_target_type=True):
        # if type discovery is turned off, infer type from pandas dtype
        if discover_target_type:
            target_type = TypeDiscoveryDriver().discover(
                target, target_series, is_target=True
            )
        else:
            target_type = get_feature_type(target, target_series)
        return target_type

    @classmethod
    def _get_dataset(
        cls,
        df: pd.DataFrame,
        sampled_df: pd.DataFrame,
        target: str,
        target_type: TypedFeature,
        shape: Tuple[int, int],
        positive_class=None,
        **init_kwargs,
    ):
        if len(df[target].dropna()) == 0:
            logger.warning(
                "It is not recommended to use an empty column as the target variable."
            )
            raise ValueError(
                f"We do not support using empty columns as the chosen target"
            )
        if is_same_class(target_type, ContinuousTypedFeature):
            return RegressionDataset(
                df=df,
                sampled_df=sampled_df,
                target=target,
                target_type=target_type,
                shape=shape,
                **init_kwargs,
            )
        elif is_same_class(
            target_type, DateTimeTypedFeature
        ) or df.index.dtype.name.startswith("datetime"):
            return ForecastingDataset(
                df=df,
                sampled_df=sampled_df,
                target=target,
                target_type=target_type,
                shape=shape,
                **init_kwargs,
            )

        # Adding ordinal typed feature, but ultimately we should rethink how we want to model this type
        elif is_same_class(target_type, CategoricalTypedFeature) or is_same_class(
            target_type, OrdinalTypedFeature
        ):
            if target_type.meta_data["internal"]["unique"] == 2:
                if is_text_data(sampled_df, target):
                    return BinaryTextClassificationDataset(
                        df=df,
                        sampled_df=sampled_df,
                        target=target,
                        shape=shape,
                        target_type=target_type,
                        positive_class=positive_class,
                        **init_kwargs,
                    )

                return BinaryClassificationDataset(
                    df=df,
                    sampled_df=sampled_df,
                    target=target,
                    shape=shape,
                    target_type=target_type,
                    positive_class=positive_class,
                    **init_kwargs,
                )
            else:
                if is_text_data(sampled_df, target):
                    return MultiClassTextClassificationDataset(
                        df=df,
                        sampled_df=sampled_df,
                        target=target,
                        target_type=target_type,
                        shape=shape,
                        **init_kwargs,
                    )
                return MultiClassClassificationDataset(
                    df=df,
                    sampled_df=sampled_df,
                    target=target,
                    target_type=target_type,
                    shape=shape,
                    **init_kwargs,
                )
        elif (
            is_same_class(target, DocumentTypedFeature)
            or "text" in target_type["type"]
            or "text" in target
        ):
            raise ValueError(
                f"The column {target} cannot be used as the target column."
            )
        elif (
            is_same_class(target_type, GISTypedFeature)
            or "coord" in target_type["type"]
            or "coord" in target
        ):
            raise ValueError(
                f"The column {target} cannot be used as the target column."
            )
        # This is to catch constant columns that are boolean. Added as a fix for pd.isnull(), and datasets with a
        #   binary target, but only data on one instance
        elif target_type["low_level_type"] == "bool":
            return BinaryClassificationDataset(
                df=df,
                sampled_df=sampled_df,
                target=target,
                shape=shape,
                target_type=target_type,
                positive_class=positive_class,
                **init_kwargs,
            )
        raise ValueError(
            f"Unable to identify problem type. Specify the data type of {target} using 'types'. "
            f"For example, types = {{{target}: 'category'}}"
        )


class CustomFormatReaders:
    @staticmethod
    def read_tsv(path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(
            path, **utils.inject_and_copy_kwargs(kwargs, **{"sep": "\t"})
        )

    @staticmethod
    def read_json(path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_json(path, **kwargs)
        except ValueError as e:
            return pd.read_json(
                path, **utils.inject_and_copy_kwargs(kwargs, **{"lines": True})
            )

    @staticmethod
    def read_libsvm(path: str, **kwargs) -> pd.DataFrame:
        from sklearn.datasets import load_svmlight_file
        from joblib import Memory

        mem = Memory("./mycache")

        @mem.cache
        def get_data(path):
            X, y = load_svmlight_file(path)
            df = pd.DataFrame(X.todense())
            df["target"] = y
            return df

        return get_data(path)

    @staticmethod
    @runtime_dependency(
        module="pandavro", object="read_avro", install_from=OptionalDependency.DATA
    )
    def read_avro(path: str, **kwargs) -> pd.DataFrame:
        return read_avro(path, **kwargs)

    DEFAULT_SQL_CHUNKSIZE = 12007
    DEFAULT_SQL_ARRAYSIZE = 50000
    DEFAULT_SQL_MIL = 128
    DEFAULT_SQL_CTU = False

    @classmethod
    def read_sql(cls, path: str, table: str = None, **kwargs) -> pd.DataFrame:
        """

        :param path: str
            This is the connection URL that gets passed to sqlalchemy's create_engine method
        :param table: str
            This is either the name of a table to select * from or a sql query to be run
        :param kwargs:
        :return: pd.DataFrame
        """
        if table is None:
            raise ValueError(
                "In order to read from a database you need to specify the table using the `table` "
                "argument."
            )
        # check if it's oracle dialect
        if str(path).lower().startswith("oracle"):
            kwargs = utils.inject_and_copy_kwargs(
                kwargs,
                **{
                    "arraysize": cls.DEFAULT_SQL_ARRAYSIZE,
                    "max_identifier_length": cls.DEFAULT_SQL_MIL,
                    "coerce_to_unicode": cls.DEFAULT_SQL_CTU,
                },
            )
        engine = utils.get_sqlalchemy_engine(path, **kwargs)

        table_name = table.strip()
        with engine.connect() as connection:
            # if it's a query expression:
            if table_name.lower().startswith("select"):
                sql_query = table_name
            else:
                sql_query = f"select * from {table_name}"

            chunks = pd.read_sql_query(
                sql_query,
                con=connection,
                **_validate_kwargs(
                    pd.read_sql_query,
                    utils.inject_and_copy_kwargs(
                        kwargs, **{"chunksize": cls.DEFAULT_SQL_CHUNKSIZE}
                    ),
                ),
            )
            df = pd.DataFrame()
            from tqdm import tqdm

            with tqdm(chunks, unit=" rows") as t:
                for chunk in chunks:
                    df = pd.concat([df, chunk])
                    t.update(len(chunk))

            df = df.reset_index(drop=True)
            if df.shape[0] == 0:
                logger.warning(
                    "The SQL expression returned zero rows. Therefore, no `ADSdataset` object was created."
                )
                raise Exception("The SQL expression returned no rows")
        return df

    @staticmethod
    def read_log(path, **kwargs):
        from ads.dataset.helper import parse_apache_log_str, parse_apache_log_datetime

        df = pd.read_csv(
            path,
            # assume_missing=True,
            sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
            engine="python",
            na_values="-",
            header=None,
            names=[
                "host",
                "identity",
                "user",
                "time",
                "request",
                "http_code",
                "response_bytes",
                "referer",
                "user_agent",
                "unknown",
            ],
            converters={
                "time": parse_apache_log_datetime,
                "request": parse_apache_log_str,
                "status": int,
                "size": int,
                "referer": parse_apache_log_str,
                "user_agent": parse_apache_log_str,
            },
            **kwargs,
        )
        return df

    @staticmethod
    def read_html(path, html_table_index: int = None, **kwargs):
        if html_table_index is None:
            return pd.concat(df for df in pd.read_html(path, **kwargs))
        else:
            return pd.read_html(path, **kwargs)[html_table_index]

    @staticmethod
    @runtime_dependency(module="scipy", install_from=OptionalDependency.VIZ)
    def read_arff(path, **kwargs):
        from scipy.io import arff
        import requests
        from io import BytesIO, TextIOWrapper

        data = None
        if os.path.isfile(path):
            data, _ = arff.loadarff(path)
        else:
            with requests.get(path) as r:
                if r.status_code == requests.codes.ok:
                    f = TextIOWrapper(BytesIO(r.content))
                    data, _ = arff.loadarff(f)
        return pd.DataFrame(data)

    @staticmethod
    def read_xml(path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from xml file.

        Parameters
        ----------
        path: str
            Path to XML file
        storage_options: dict, optional
            Storage options passed to Pandas to read the file.

        Returns
        -------
        dataframe : pandas.DataFrame
        """
        import xml.etree.cElementTree as et

        def get_children(df, node, parent, i):

            for name in node.attrib.keys():
                df.at[i, parent + name] = node.attrib[name]
            for child in list(node):
                if len(list(child)) > 0:
                    get_children(df, child, parent + child.tag + "/", i)
                else:
                    df.at[i, parent + child.tag] = child.text

        storage_options = kwargs.get("storage_options", {})

        file_handles = fsspec.open_files(path, mode="rb", **storage_options)
        ret_df = pd.DataFrame()
        last_i = 0
        for file_handle in file_handles:
            with file_handle:
                parsed_xml = et.parse(path)
                for i, node in enumerate(parsed_xml.getroot()):
                    get_children(ret_df, node, node.tag + "/", last_i + i)
                last_i = i
        return ret_df


reader_fns = {
    "csv": pd.read_csv,
    "tsv": CustomFormatReaders.read_tsv,
    "json": CustomFormatReaders.read_json,
    "jsonl": CustomFormatReaders.read_json,
    "excel": pd.read_excel,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "parquet": pd.read_parquet,
    "libsvm": CustomFormatReaders.read_libsvm,
    "hdf": pd.read_hdf,  # Todo: re.match(format, "hdf\d*") or format == "h5"
    "hdf3": pd.read_hdf,
    "hdf4": pd.read_hdf,
    "h5": pd.read_hdf,
    "avro": CustomFormatReaders.read_avro,
    "avsc": CustomFormatReaders.read_avro,
    "sql": CustomFormatReaders.read_sql,
    "db": CustomFormatReaders.read_sql,
    "log": CustomFormatReaders.read_log,
    "clf": CustomFormatReaders.read_log,
    "html": CustomFormatReaders.read_html,
    "arff": CustomFormatReaders.read_arff,
    "xml": CustomFormatReaders.read_xml,
}


def _validate_kwargs(func: Callable, kwargs):
    valid_params = inspect.signature(func).parameters
    if "kwargs" in valid_params:
        return kwargs
    else:
        return {k: v for k, v in kwargs.items() if k in valid_params}


def get_format_reader(path: ElaboratedPath, **kwargs) -> Callable:
    format_key = path.format
    try:
        reader_fn = reader_fns[format_key]
    except (KeyError, NameError):
        raise ValueError(
            f"We were unable to load the specified dataset. We have interpreted the format "
            f"as {format_key}, if this is not correct, call again and set the `format` parameter = "
            f"to the desired format. Read more here: https://docs.cloud.oracle.com/en-us/iaas/tools/ads"
            f"-sdk/latest/user_guide/loading_data/loading_data.html#specify-data-types-in-load-dataset"
        )

    return reader_fn


def load_dataset(path: ElaboratedPath, reader_fn: Callable, **kwargs) -> pd.DataFrame:
    dfs = []
    for filename in path.paths:
        data = reader_fn(filename, **_validate_kwargs(reader_fn, kwargs))
        if not isinstance(data, pd.DataFrame):
            fn_name = f"{reader_fn.__module__}.{reader_fn.__qualname__}"
            raise ValueError(
                f"{fn_name} is used to load the data. "
                f"However, {fn_name} returned {type(data)} instead of pandas DataFrame. "
                f"Refer to the usage of {fn_name} to set the correct arguments."
            )
        dfs.append(data)
    if len(dfs) == 0:
        raise ValueError(
            f"We were unable to load the specified dataset. Read more here: "
            f"https://docs.cloud.oracle.com/en-us/iaas/tools/ads"
            f"-sdk/latest/user_guide/loading_data/loading_data.html#specify-data-types-in-load-dataset"
        )

    df = pd.concat(dfs)

    if df is None:
        raise ValueError(
            f"We were unable to load the specified dataset. Read more here: "
            f"https://docs.cloud.oracle.com/en-us/iaas/tools/ads"
            f"-sdk/latest/user_guide/loading_data/loading_data.html#specify-data-types-in-load-dataset"
        )
    if df.empty:
        raise DatasetLoadException("Empty DataFrame, not producing a ADSDataset")
    return df
