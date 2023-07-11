#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ast
import base64
import html
import io
import math
import os
import warnings
import re
from collections import defaultdict
import inspect
import importlib
from typing import Callable, List, Tuple, Union
import fsspec

# from pandas.io.common import _compression_to_extension

from numbers import Number
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from pandas.core.dtypes.common import (
    is_numeric_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
)

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import utils
from ads.dataset import logger
from ads.type_discovery.type_discovery_driver import TypeDiscoveryDriver
from ads.type_discovery.typed_feature import (
    ContinuousTypedFeature,
    DateTimeTypedFeature,
    CategoricalTypedFeature,
    GISTypedFeature,
    TypedFeature,
    UnknownTypedFeature,
    OrdinalTypedFeature,
    DocumentTypedFeature,
)


class DatasetDefaults:
    sampling_confidence_level = 95
    sampling_confidence_interval = 1.0


_known_db_protocols = {"sqlite", "ADB", "oracle+cx_oracle"}


def concatenate(X, y):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return pd.concat([X, y], axis=1)
    else:
        return X.assign(**{y.name: y})


def fix_column_names(X):
    X.columns = X.columns.astype("str").str.strip().str.replace(" ", "_")
    return X


def convert_columns(df, feature_metadata=None, dtypes=None):
    if feature_metadata is not None:
        dtypes = {}
        for feature in feature_metadata:
            dtype = get_dtype(feature_metadata[feature], df[feature].dtype)
            if dtype is not None:
                dtypes[feature] = dtype
    return df.astype(dtypes)


def get_dtype(feature_type, dtype):
    if isinstance(feature_type, ContinuousTypedFeature) or isinstance(
        feature_type, OrdinalTypedFeature
    ):
        return dtype.name if is_numeric_dtype(dtype) else "float"
    elif isinstance(feature_type, DateTimeTypedFeature):
        return "datetime64[ns]" if not dtype.name.startswith("datetime") else dtype
    elif isinstance(feature_type, CategoricalTypedFeature):
        return "bool" if is_bool_dtype(dtype) else "category"


def get_feature_type(name, series):
    if is_bool_dtype(series) or is_categorical_dtype(series):
        return CategoricalTypedFeature.build(name, series)
    elif is_numeric_dtype(series):
        if is_float_dtype(series):
            return ContinuousTypedFeature.build(name, series)
        else:
            return OrdinalTypedFeature.build(name, series)
    elif is_datetime64_any_dtype(series):
        return DateTimeTypedFeature.build(name, series)
    else:
        return UnknownTypedFeature.build(name, series)


def convert_to_html(plot):
    img = io.BytesIO()
    plot.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    return '<img width=95%" src="data:image/png;base64, {}"><hr><br>'.format(
        encoded.decode("utf-8")
    )


def _num_partitions_for_dataframe(df):
    # takes pandas dataframe, guesses good number of partitions
    return utils.get_cpu_count() if df.shape[0] > 1000 * utils.get_cpu_count() else 1


class ElaboratedPath:
    """
    The Elaborated Path class unifies all of the operations and information related to a path or pathlist.
    Whether the user wants to
    An Elaborated path can accept any of the following as a valid source:
    * A single path
    * A glob pattern path
    * A directory
    * A list of paths (Note: all of these paths must be from the same filesystem AND have the same format)
    * A sqlalchemy connection url
    """

    def __init__(
        self,
        source: Union[str, List[str]],
        format: str = None,
        name: str = None,
        **kwargs,
    ):
        """
        :param source:
        :param format:
        :param kwargs:

        By the end of this method, this class needs to have paths, format, and name ready
        """
        self._kwargs = kwargs
        self._format = format
        self._name = name
        if isinstance(source, str):
            self._original_source = source
            self._determine_protocol_type()
            if self._type == "db":
                self._paths = [self._original_source]
            else:
                self._elaborate_path()
        elif isinstance(source, list) and all(isinstance(file, str) for file in source):
            assert len(source) > 0, "Error, the source you passed in was an empty list."
            self._original_source = source[0]
            self._paths = source
            self._type = "list"
        else:
            raise ValueError(f"Source argument not understood: {source}")
        if self.num_paths == 0:
            raise FileNotFoundError(
                f"Error: We could not find any files associated with the source: "
                f"{source}. Double check that this source is a valid glob pattern,"
                f" directory, or path."
            )
        self._determine_format()
        self._determine_name()

    @property
    def paths(self) -> List[str]:
        """
        :return: a list of str
            Each element will be a valid path
        """
        return self._paths

    @property
    def num_paths(self) -> int:
        """
        This method will return the number of paths found with the associated original glob, folder, or path.
        If this returns 0,
        :return:
        """
        return len(self._paths)

    @property
    def name(self) -> str:
        return self._name

    @property
    def format(self) -> str:
        return self._format

    def _determine_name(self):
        if self._name is None:
            if self._type == "list":
                self._name = (
                    f"DataFrame from [{os.path.basename(self._original_source)}, ...]"
                )
            elif self._type == "glob":
                self._name = f"DataFrame from {os.path.basename(self._original_source)}"
            else:
                self._name = f"DataFrame from {urlparse(self._original_source).scheme}"

    def _determine_format(self):
        """
        Infer format from the path.

        If its  a compressed file, returns the extension before compression extension.
        If the extension cannot be inferred,  returns None

        Parameters
        ----------
        path : ElaboratedPath
            an ElaboratedPath object

        Returns
        -------
        format : str
        """
        if self._format in [None, "infer"]:
            format_keys = []
            for i in range(min(self.num_paths, 5)):
                format_keys.append(self._remove_compressions(self.paths[i]))
            if len(format_keys) == 0:
                raise ValueError(
                    f"Could not determine the format key for source: {self._original_source}"
                )
            if format_keys.count(format_keys[0]) != len(format_keys):
                raise ValueError(
                    f"Got multiple formats from the source: {self._original_source}. Run again "
                    f'using the format parameter. Ex: format=<your format key, like: "csv", "hdf", etc.>'
                )
            self._format = format_keys[0]
        else:
            self._format = self._format.lower()

    def _elaborate_path(self):
        self._paths = self._fs.glob(self._original_source)
        if self._protocol != "":
            self._paths = [f"{self._protocol}://{p}" for p in self._paths]

    def _determine_protocol_type(self):
        self._protocol = urlparse(self._original_source).scheme

        if self._kwargs.get("fs") is not None:
            self._fs = self._kwargs.pop("fs")
            self._type = "glob"
        elif self._original_source.startswith("oracle+cx_oracle://"):
            self._protocol = "oracle+cx_oracle"
            self._type = "db"
        else:
            try:
                self._fs = fsspec.filesystem(
                    self._protocol, **self._kwargs.get("storage_options", dict())
                )
                self._type = "glob"
            except ValueError:
                try:
                    self.engine = utils.get_sqlalchemy_engine(
                        self._original_source, **self._kwargs
                    )
                    self._type = "db"
                except:
                    if self._protocol in _known_db_protocols:
                        self._type = "db"
                    else:
                        raise ValueError(
                            f"Error in trying to understand the protocol for source: "
                            f"{self._original_source}. The protocol found: {self._protocol} is not "
                            f"registered with fsspec or sqlalchemy"
                        )

    @staticmethod
    def _remove_compressions(filename: str):
        _compression_to_extension = [
            ".gz",
            ".bz2",
            ".zip",
            ".xz",
            ".zst",
            ".tar",
            ".tar.gz",
            ".tar.xz",
            ".tar.bz2",
        ]
        for compression in _compression_to_extension:
            if filename.strip().endswith(compression):
                return ElaboratedPath._remove_compressions(
                    os.path.splitext(filename.rstrip("/*"))[0]
                )
        format = os.path.splitext(filename.rstrip("/*"))[1][1:].lower()
        return format.lower() if format != "" else None


class DatasetLoadException(BaseException):
    def __init__(self, exc_msg):
        self.exc_msg = exc_msg

    def __str__(self):
        return self.exc_msg


def _get_dtype_from_error(e):
    error_string = str(e)

    if "mismatched dtypes" in error_string.lower():

        # For the mismatched dtypes error, dask either returns a error message containing the dtype argument
        # to specify, or  the found and expected dtypes in a table format, depending on what stage
        # the type inferencing fails. The below logic supports building the dtype dictionary for both cases
        found_dtype_dict_str_list = re.findall(
            r"dtype=({[^{}]+})", error_string, re.MULTILINE
        )
        if found_dtype_dict_str_list:
            found_dtype_dict = ast.literal_eval(found_dtype_dict_str_list[0])
        else:
            found_dtype_dict = _find_dtypes_from_table(error_string)
        if found_dtype_dict:
            logger.warning(
                "Dask type-inference/coercion failed. Retrying with "
                f"dtype={found_dtype_dict}.",
                exc_info=True,
            )
            return found_dtype_dict
    return None


def _find_dtypes_from_table(error_string):
    error_lines = error_string.splitlines()
    dtypes = {}
    # matches '| Column | Found | Expected |'
    pattern = re.compile(
        "\\s*\\|\\s*Column\\s*\\|\\s*Found\\s*\\|\\s*Expected\\s*\\|\\s*"
    )
    for i, line in enumerate(error_lines):
        if re.match(pattern, line):
            for j in range(i + 2, len(error_lines)):
                # extracts column_name and found_dtype from  '| <column_name> | <found_dtype> | <expected_dtype |'
                dtype_suggestion = re.compile("\\s*\\|([^\\|]+)\\|([^\\|]+)\\|.*")
                match_groups = re.match(dtype_suggestion, error_lines[j])
                if match_groups is None:
                    break
                dtypes[match_groups.group(1).strip()] = match_groups.group(2).strip()
    return dtypes


def rename_duplicate_cols(original_cols):
    seen_col_names = defaultdict(int)
    new_cols = []
    for col in original_cols:
        # remove any spaces form column names
        if isinstance(col, str):
            col.replace(" ", "_")
        if col not in seen_col_names:
            new_cols.append(col)
        else:
            dup_count = seen_col_names[col]
            new_cols.append(f"{col}.{dup_count}")
        seen_col_names[col] += 1
    assert len(new_cols) == len(
        original_cols
    ), "There has been an error in re-naming duplicate columns"
    return new_cols


def write_parquet(
    path,
    data,
    engine="fastparquet",
    metadata_dict=None,
    compression=None,
    storage_options=None,
):
    """
    Uses fast parquet to write dask dataframe and custom metadata in parquet format

    Parameters
    ----------
    path : str
        Path to write to
    data : pandas.DataFrame
    engine : string
        "auto" by default
    metadata_dict : Deprecated, will not pass through
    compression : {{'snappy', 'gzip', 'brotli', None}}, default 'snappy'
        Name of the compression to use
    storage_options : dict, optional
        storage arguments required to read the path

    Returns
    -------
    str : the file path the parquet was written to
    """
    assert isinstance(data, pd.DataFrame)
    if metadata_dict is not None:
        warnings.warn(
            "The `metadata_dict` argument is deprecated and has no effect on this method.",
            DeprecationWarning,
            stacklevel=2,
        )
    data.to_parquet(
        path,
        engine=engine,
        compression=compression,
        storage_options=storage_options,
    )
    return path


def is_text_data(df, target=None):
    if len(df.columns.values) == 2:
        feature_name = (
            list(set(df.columns.values) ^ set([target]))[0]
            if target
            else list(set(df.columns.values))[0]
        )
    elif len(df.columns.values == 1):
        feature_name = df.columns.values[0]
    else:
        return False
    return isinstance(
        TypeDiscoveryDriver().discover(feature_name, df[feature_name]),
        DocumentTypedFeature,
    )


def generate_sample(
    df: pd.DataFrame,
    n: int,
    confidence_level: int = DatasetDefaults.sampling_confidence_level,
    confidence_interval: float = DatasetDefaults.sampling_confidence_interval,
    **kwargs,
):
    min_size_to_sample = min(n, 10000)

    sample_size = None

    if "sample_max_rows" in kwargs:
        requested_sample_size = int(kwargs["sample_max_rows"])

        if requested_sample_size < 0:
            sample_size = calculate_sample_size(
                n, min_size_to_sample, confidence_level, confidence_interval
            )
        else:
            if min_size_to_sample < requested_sample_size < n:
                logger.info(
                    f"Downsampling from {n} rows, to the user specified {requested_sample_size} rows for graphing."
                )
                sample_size = requested_sample_size
            elif requested_sample_size >= n:
                logger.info(f"Using the entire dataset of {n} rows for graphing.")
                sample_size = n
            else:
                sample_size = min_size_to_sample
                logger.info(
                    f"Downsampling from {n} rows, to {sample_size} rows for graphing."
                )

    if sample_size and len(df) > sample_size:
        frac = min(1.0, sample_size * 1.05 / n)
        df = df.sample(frac=frac, random_state=42)
        return df.head(sample_size) if len(df) > sample_size else df
    else:
        return df


def calculate_sample_size(
    population_size, min_size_to_sample, confidence_level=95, confidence_interval=1.0
):
    """Find sample size for a population using Cochran’s Sample Size Formula.
     With default values for confidence_level (percentage, default: 95%)
     and confidence_interval (margin of error, percentage, default: 1%)

    SUPPORTED CONFIDENCE LEVELS: 50%, 68%, 90%, 95%, and 99% *ONLY* - this
    is because the Z-score is table based, and I'm only providing Z
    for common confidence levels.
    """

    if population_size < min_size_to_sample:
        return None

    confidence_level_constant = {
        50: 0.67,
        68: 0.99,
        90: 1.64,
        95: 1.96,
        99: 2.57,
        99.5: 2.807,
        99.9: 3.291,
    }

    p = 0.5
    e = confidence_interval / 100.0
    N = population_size
    n_0 = 0.0
    n = 0.0

    Z = confidence_level_constant.get(confidence_level, 99)

    n_0 = ((Z**2) * p * (1 - p)) / (e**2)
    n = n_0 / (1 + ((n_0 - 1) / float(N)))

    sample_size = max(int(math.ceil(n)), min_size_to_sample)

    logger.info(f"Downsampling from {population_size} rows to {sample_size} rows.")

    return sample_size


def map_types(types):
    for column in types:
        if types[column] == "continuous":
            types[column] = "float64"
        elif types[column] == "ordinal":
            types[column] = "int64"
        elif types[column] == "categorical":
            types[column] = "category"
        elif types[column] == "datetime":
            types[column] = "datetime64[ns]"
    return types


@runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
@runtime_dependency(module="graphviz", install_from=OptionalDependency.VIZ)
def visualize_transformation(transformer_pipeline, text=None):
    dot = graphviz.Digraph()

    # show a single node for paritions
    dot.attr(
        "node",
        shape="tab",
        style="filled",
        fontname="courier",
        fontsize="12",
        fontcolor="white",
        resolution="144",
    )
    if text:
        dot.node("partitions", text, margin="0.25", fillcolor="dimgray")

    dot.attr(
        "node",
        shape="component",
        style="filled",
        fontname="courier",
        fontsize="10",
        fontcolor="black",
        resolution="144",
    )
    for step in transformer_pipeline.steps:
        name, clazz, clazzname, is_ads = (
            step[0],
            step[1],
            step[1].__class__.__name__,
            "ads" in str(step[1].__class__),
        )
        ads_node = str(step[1].__class__.__name__) in [
            "AutoMLPreprocessingTransformer",
            "DataFrameTransformer",
            "RecommendationTransformer",
            "AutoMLFeatureSelection",
            "FeatureEngineeringTransformer",
        ]
        if ads_node:
            text2html = "< {} >".format(
                html.escape(step[1].__repr__()).replace("\n", "<br/>")
            )
            dot.node(name, text2html, margin="0.25", fillcolor="gold2")
        else:
            dot.node(name, name.rsplit("/")[0], fillcolor="azure")

    def format_label(stage):
        if "FunctionTransformer" in str(transformer_pipeline.steps[stage][1].__class__):
            return "<<font face='courier' point-size='10'>&nbsp;<b>{}</b>&nbsp;</font>>".format(
                html.escape(transformer_pipeline.steps[stage][1].func.__name__)
            )
        else:
            is_ads = "ads" in str(transformer_pipeline.steps[stage][1].__class__)
            return "<<font face='courier' point-size='10'>&nbsp;<b>{}</b>&nbsp;</font>>".format(
                transformer_pipeline.steps[stage][1].__class__.__name__
            )

    edges = [x[0] for x in transformer_pipeline.steps]
    for i, edge in enumerate(list(zip(edges[:-1], edges[1:]))):
        dot.edge(*edge, len="1.00", label=format_label(i))

    # terminus node
    dot.node("terminus", "", shape="terminator", fillcolor="white")
    dot.edge(edges[-1], "terminus", len="1.00", label=format_label(len(edges) - 1))

    graph = graphviz.Source(dot)

    from IPython.core.display import display, SVG

    display(SVG(graph.pipe(format="svg")))


def up_sample(df, target, sampler="default", feature_types=None):
    """
    Fixes imbalanced dataset by up-sampling

    Parameters
    ----------
    df : Union[pandas.DataFrame, dask.dataframe.core.DataFrame]
    target : name of the target column in df
    sampler: Should implement fit_resample(X,y) method
    fillna: a dictionary contains the column name as well as the fill value,
            only needed when the column has missing values

    Returns
    -------
    upsampled_df : Union[pandas.DataFrame, dask.dataframe.core.DataFrame]
    """
    if sampler != "default":
        if inspect.getattr_static(sampler, "fit_resample", None) is None:
            raise AttributeError("`sampler` object must has method `fit_resample`.")
        else:
            # exactly two input args X, y will be passed to fit_resample()
            # check signature of fit_sample
            num_no_default_params = 0
            sig = inspect.signature(sampler.fit_resample)
            for param in sig.parameters.values():
                if param.default is param.empty:
                    num_no_default_params += 1
            if len(sig.parameters) < 2 or num_no_default_params > 2:
                raise RuntimeError(
                    "The signature for `sampler.fit_resample` has to be `fit_resample(X, y)`."
                )

    X = df.drop(target, axis=1)
    y = df[target]

    feature_types = feature_types if feature_types is not None else {}

    columns_with_nans = X.columns.values[X.isna().any()]
    if len(columns_with_nans) > 0:
        fill_nan_dict = {}
        for column in columns_with_nans:
            if column in feature_types and "mode" in feature_types[column]["stats"]:
                fill_nan_dict[column] = feature_types[column]["stats"]["mode"]
            elif column in feature_types and "mean" in feature_types[column]["stats"]:
                fill_nan_dict[column] = feature_types[column]["stats"]["mean"]
            elif column in feature_types and "median" in feature_types[column]["stats"]:
                fill_nan_dict[column] = feature_types[column]["stats"]["median"]
            else:
                logger.warning(
                    "Sampling from a column that has missing values may cause an error."
                )
        X = X.fillna(fill_nan_dict)

    if sampler == "default":
        imblearn_found = importlib.util.find_spec("imblearn") is not None
        if not imblearn_found:
            raise ModuleNotFoundError(
                """
                Required package for up-sampling `imblearn` not found.
                Install `imblearn` with `pip install imbalanced-learn`
                and rerun to enable up-sampling.
            """
            )
        else:
            sampler = _get_imblearn_sampler(X, y)
    return _sample(sampler, X, y)


def _get_imblearn_sampler(X, y):
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    categorical_feature_indices = [
        X.columns.get_loc(c)
        for c in X.select_dtypes(
            include=["category", "object", "datetime64"]
        ).columns.values
    ]

    if len(categorical_feature_indices) > 0:
        logger.info(
            """
            Using the default `RandomOverSampler` sampler. Use `sample` to specify a sampler.
            Classes will be equalized.
            You can also pass in other samplers such as `imblearn.SMOTENC` instead, e.g.
            sampler = SMOTENC(categorical_features=categorical_feature_indices)
            ds.up_sample(sampler=sampler)
        """
        )
        return RandomOverSampler(random_state=42)

    min_sample_size = y.value_counts().min()

    k_neighbors = min(min_sample_size - 1, 5)
    if k_neighbors == 0:
        logger.warning(
            f"""k_neighbors is 0 as in the target there exists a class label that appeared only once.
                SMOTE will fail. Default to RandomOverSampler.
            """
        )
        return RandomOverSampler(random_state=42)
    else:
        if 5 > k_neighbors > 0:
            logger.info(
                f"`k_neighbors()` of SMOTE has changed to {k_neighbors}"
                " as the target has at least one class which appeared "
                f"only {min_sample_size} times in the data. "
            )
        logger.info("Using SMOTE for over sampling. Classes will be equalized.")
        return SMOTE(random_state=42, k_neighbors=k_neighbors)


def down_sample(df, target):
    """
    Fixes imbalanced dataset by down-sampling

    Parameters
    ----------
    df : pandas.DataFrame
    target : name of the target column in df

    Returns
    -------
    downsampled_df : pandas.DataFrame
    """
    dfs = []
    target_value_counts = df[target].value_counts()
    min_key = min(target_value_counts.iteritems(), key=lambda k: k[1])
    for key, value in target_value_counts.iteritems():
        if key != min_key[0]:
            dfs.append(
                df[df[target] == key].sample(frac=1 - ((value - min_key[1]) / value))
            )
    dfs.append(df[df[target] == min_key[0]])
    return pd.concat(dfs)


def _sample(sampler, X, y):
    if isinstance(y, pd.Series) and (
        isinstance(y[0], bool) or isinstance(y[0], np.bool_)
    ):
        y_trans = y.astype(int)  ## Convert to ints to let SMOTE sample properly
        X_resampled, y_resampled = sampler.fit_resample(X=X, y=y_trans)
    else:
        X_resampled, y_resampled = sampler.fit_resample(X=X, y=y)

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns.values)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.DataFrame(y_resampled, columns=[y.name])[y.name]

    for k in X.dtypes.keys():
        X_resampled[k] = X_resampled[k].astype(X.dtypes[k].name)
    balanced_df = concatenate(X_resampled, y_resampled)
    return balanced_df


def get_fill_val(feature_types, column, action, constant="constant"):
    # action can be one of the following
    # "Fill missing values with mean", "Fill missing values with median",
    # "Fill missing values with frequent", "Fill missing values with constant"
    action_ = action.split(" ")[-1]
    fill_type = "mode" if action_ == "frequent" else action_
    try:
        fill_val = (
            feature_types[column].meta_data["stats"][fill_type]
            if action_ != "constant"
            else constant
        )
        fill_val = round(fill_val, 4) if isinstance(fill_val, Number) else fill_val
    except:
        fill_val = None
    return fill_val


def parse_apache_log_str(x):
    """
    Returns the string delimited by two characters.

    Source: https://mmas.github.io/read-apache-access-log-pandas
    Example:
        `>>> parse_str('[my string]')`
        `'my string'`
    """
    if x is not None:
        return x[1:-1]
    return np.nan


def parse_apache_log_datetime(x):
    """
    Parses datetime with timezone formatted as:
        `[day/month/year:hour:minute:second zone]`

    Source: https://mmas.github.io/read-apache-access-log-pandas
    Example:
        `>>> parse_datetime('13/Nov/2015:11:45:42 +0000')`
        `datetime.datetime(2015, 11, 3, 11, 45, 4, tzinfo=<UTC>)`

    Due to problems parsing the timezone (`%z`) with `datetime.strptime`, the
    timezone will be obtained using the `pytz` library.
    """
    import pytz
    from datetime import datetime

    dt = datetime.strptime(x[1:-7], "%d/%b/%Y:%H:%M:%S")
    dt_tz = int(x[-6:-3]) * 60 + int(x[-3:-1])
    return dt.replace(tzinfo=pytz.FixedOffset(dt_tz))


def deprecate_variable(old_var, new_var, warning_msg, warning_type):
    if old_var is not None:
        warnings.warn(warning_msg, warning_type)
        return old_var
    return new_var


def deprecate_default_value(var, old_value, new_value, warning_msg, warning_type):
    if var == old_value:
        warnings.warn(warning_msg, warning_type)
        return new_value
    else:
        return var


def _log_yscale_not_set():
    logger.info(
        "`yscale` parameter is not set. Valid values are `'linear'`, `'log'`, `'symlog'`."
    )

def infer_target_type(target, target_series, discover_target_type=True):
    # if type discovery is turned off, infer type from pandas dtype
    if discover_target_type:
        target_type = TypeDiscoveryDriver().discover(
            target, target_series, is_target=True
        )
    else:
        target_type = get_feature_type(target, target_series)
    return target_type

def get_target_type(target, sampled_df, **init_kwargs):
    discover_target_type = init_kwargs.get("type_discovery", True)
    if target in init_kwargs.get("types", {}):
        sampled_df[target] = sampled_df[target].astype(init_kwargs.get("types")[target])
        discover_target_type = False
    return infer_target_type(target, sampled_df[target], discover_target_type)

def get_dataset(
    df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    target: str,
    target_type: TypedFeature,
    shape: Tuple[int, int],
    positive_class=None,
    **init_kwargs,
):  
    from ads.dataset.classification_dataset import (
        BinaryClassificationDataset, 
        BinaryTextClassificationDataset, 
        MultiClassClassificationDataset, 
        MultiClassTextClassificationDataset
    )
    from ads.dataset.forecasting_dataset import ForecastingDataset
    from ads.dataset.regression_dataset import RegressionDataset

    if len(df[target].dropna()) == 0:
        logger.warning(
            "It is not recommended to use an empty column as the target variable."
        )
        raise ValueError(
            f"We do not support using empty columns as the chosen target"
        )
    if utils.is_same_class(target_type, ContinuousTypedFeature):
        return RegressionDataset(
            df=df,
            sampled_df=sampled_df,
            target=target,
            target_type=target_type,
            shape=shape,
            **init_kwargs,
        )
    elif utils.is_same_class(
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
    elif utils.is_same_class(target_type, CategoricalTypedFeature) or utils.is_same_class(
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
        utils.is_same_class(target, DocumentTypedFeature)
        or "text" in target_type["type"]
        or "text" in target
    ):
        raise ValueError(
            f"The column {target} cannot be used as the target column."
        )
    elif (
        utils.is_same_class(target_type, GISTypedFeature)
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
    return build_dataset(
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

def build_dataset(
    df: pd.DataFrame,
    shape: Tuple[int, int],
    target: str = None,
    progress=None,
    **kwargs,
):
    from ads.dataset.dataset import ADSDataset

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
    target_type = infer_target_type(
        target, sampled_df[target], discover_target_type
    )

    result = get_dataset(
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
                **validate_kwargs(
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

def validate_kwargs(func: Callable, kwargs):
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
        data = reader_fn(filename, **validate_kwargs(reader_fn, kwargs))
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
