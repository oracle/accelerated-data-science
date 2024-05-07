#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import absolute_import, print_function

import collections
import contextlib
import copy
import fnmatch
import glob
import json
import math
import os
import random
import re
import shutil
import string
import sys
import tempfile
from datetime import datetime
from enum import Enum
from io import DEFAULT_BUFFER_SIZE
from pathlib import Path
from textwrap import fill
from typing import Dict, Optional, Union
from urllib import request
from urllib.parse import urlparse

import fsspec
import matplotlib as mpl
import numpy as np
import pandas as pd
from cycler import cycler
from oci import object_storage
from pandas.core.dtypes.common import is_datetime64_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ads import config
from ads.common import logger
from ads.common.decorator.deprecate import deprecated
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_client import OCIClientFactory
from ads.common.word_lists import adjectives, animals
from ads.dataset.progress import TqdmProgressBar

from . import auth as authutil

# For Model / Model Artifact libraries
lib_translator = {"sklearn": "scikit-learn"}
module_ignore = ["builtins", "ads", "automl", "mlx"]

# up-sample if length of dataframe is less than or equal to MAX_LEN_FOR_UP_SAMPLING
MAX_LEN_FOR_UP_SAMPLING = 5000

# down-sample if ratio of minority to majority class is greater than or equal to MIN_RATIO_FOR_DOWN_SAMPLING
MIN_RATIO_FOR_DOWN_SAMPLING = 1 / 20

# Maximum distinct values by cardinality will be used for plotting
MAX_DISPLAY_VALUES = 10

# par link of the index json file.
PAR_LINK = "https://objectstorage.us-ashburn-1.oraclecloud.com/p/WyjtfVIG0uda-P3-2FmAfwaLlXYQZbvPZmfX1qg0-sbkwEQO6jpwabGr2hMDBmBp/n/ociodscdev/b/service-conda-packs/o/service_pack/index.json"

random_state = 42
test_size = 0.3
date_format = "%Y-%m-%d %H:%M:%S"

# at this time, we only support regression and classification tasks.
ml_task_types = Enum(
    "ml_task_types",
    "REGRESSION BINARY_CLASSIFICATION MULTI_CLASS_CLASSIFICATION BINARY_TEXT_CLASSIFICATION "
    "MULTI_CLASS_TEXT_CLASSIFICATION UNSUPPORTED",
)

mpl.rcParams["image.cmap"] = "BuGn"
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=["teal", "blueviolet", "forestgreen", "peru", "y", "dodgerblue", "r"]
)

# sqlalchemy engines
_engines = {}

ORACLE_DEFAULT_PORT = 1521
MYSQL_DEFAULT_PORT = "3306"

# Maximum number of columns of data to extract model schema.
DATA_SCHEMA_MAX_COL_NUM = 2000

# dimention of np array which can be converted to pd dataframe
DIMENSION = 2

# declare custom exception class

# The number of worker processes to use in parallel for uploading individual parts of a multipart upload.
DEFAULT_PARALLEL_PROCESS_COUNT = 9

LOG_LEVELS = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class FileOverwriteError(Exception):  # pragma: no cover
    pass


def get_cpu_count():
    """
    Returns the number of CPUs available on this machine
    """
    return os.cpu_count()


@deprecated(
    "2.5.2", details="This method is being deprecated in favor of `get_cpu_count`"
)
def get_compute_accelerator_ncores():
    return get_cpu_count()


@deprecated(
    "2.5.10",
    details="Deprecated, use: from ads.common.auth import AuthState;"
    "oci_config_location=AuthState().oci_config_path; profile=AuthState().oci_key_profile",
)
def get_oci_config():
    """
    Returns the OCI config location, and the OCI config profile.
    """
    oci_config_location = os.environ.get(
        "OCI_CONFIG_LOCATION", f"{os.environ['HOME']}/.oci/config"
    )  # os.environ['HOME'] == home/datascience
    oci_config_profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    return oci_config_location, oci_config_profile


@deprecated(
    "2.5.10",
    details="Deprecated, use: from ads.common.auth import AuthState; os.path.dirname(AuthState().oci_config_path)",
)
def oci_key_location():
    """
    Returns the OCI key location
    """
    return os.environ.get(
        "OCI_CONFIG_DIR", os.path.join(os.path.expanduser("~"), ".oci")
    )


@deprecated(
    "2.5.10",
    details="Deprecated, use: from ads.common.auth import AuthState; AuthState().oci_config_path",
)
def oci_config_file():
    """
    Returns the OCI config file location
    """
    return os.path.join(oci_key_location(), "config")


@deprecated(
    "2.5.10",
    details="Deprecated, use: from ads.common.auth import AuthState; AuthState().oci_key_profile",
)
def oci_config_profile():
    """
    Returns the OCI config profile location.
    """
    return os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")


def numeric_pandas_dtypes():
    """
    Returns a list of the "numeric" pandas data types
    """
    return ["int16", "int32", "int64", "float16", "float32", "float64"]


@deprecated(
    "2.5.10",
    details="Deprecated, use: ads.set_auth(auth='api_key', oci_config_location='~/.oci/config', profile='DEFAULT')",
)
def set_oci_config(oci_config_location, oci_config_profile):
    """
    :param oci_config_location: location of the config file, for example, ~/.oci/config
    :param oci_config_profile: The profile to load from the config file.  Defaults to "DEFAULT"
    """
    if not os.path.exists(f"{oci_config_location}"):
        raise ValueError("The oci config file doesn't exist.")
    os.environ["OCI_CONFIG_LOCATION"] = oci_config_location
    os.environ["OCI_CONFIG_PROFILE"] = oci_config_profile


def random_valid_ocid(prefix="ocid1.dataflowapplication.oc1.iad"):
    """
    Generates a random valid ocid.

    Parameters
    ----------
    prefix: `str`
        A prefix, corresponding to a region location.

    Returns
    -------
    ocid: `str`
        a valid ocid with the given prefix.
    """
    left, right = prefix.rsplit(".", 1)
    fake = "".join([random.choice(string.ascii_lowercase) for i in range(60)])
    return f"{left}.{fake}"


def get_dataframe_styles(max_width=75):
    """Styles used for dataframe, example usage:

        df.style\
            .set_table_styles(utils.get_dataframe_styles())\
            .set_table_attributes('class=table')\
            .render())

    Returns
    -------
    styles: array
        A list of dataframe table styler styles.
    """

    alt_props = [
        ("background-color", "#F8F8F8"),
    ]

    th_props = [
        ("font-size", "12px"),
        ("text-align", "center"),
        ("font-weight", "bold"),
        ("background-color", "#D3D3D3"),
        ("padding-left", "5px"),
        ("padding-right", "5px"),
        ("text-align", "right"),
    ]

    td_props = [
        ("font-size", "12px"),
        ("text_wrap", False),
        ("white-space", "nowrap"),
        ("overflow", "hidden"),
        ("text-overflow", "ellipsis"),
        ("max-width", f"{max_width}px"),
    ]

    hover_props = [("background-color", "#D9EDFD")]

    styles = [
        dict(selector="tbody tr:nth-child(even)", props=alt_props),
        dict(selector="tbody tr:hover", props=hover_props),
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props),
    ]

    return styles


def get_bootstrap_styles():
    """
    Returns HTML bootstrap style information
    """
    return """<style>

        code {
            padding: 2px 4px;
            font-size: 90%;
            color: #c7254e;
            background-color: #f9f2f4;
            border-radius: 4px;
            font-family: Menlo,Monaco,Consolas,"Courier New",monospace;
        }

        .label {
            display: inline;
            padding: .2em .6em .3em;
            font-weight: 700;
            line-height: 1;
            color: #fff;
            font-size: 85%;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: .25em;
        }

        .label-high-cardinality {
            background-color: #fe7c1b;
        }

        .label-missing {
            background-color: #214761;
        }

        .label-zeros {
            background-color: #333796;
        }

        .label-warning {
            background-color: #e2007e;
        }

        .label-skew {
            background-color: #ffdb58;
            color: black;
        }

        .label-duplicate-rows {
            background-color: #d90773;
        }
    </style>"""


def highlight_text(text):
    """Returns text with html highlights.
    Parameters
    ----------
    text: String
      The text to be highlighted.

    Returns
    -------
    ht: String
        The text with html highlight information.
    """
    return f"""<code style="background:yellow; color:black; padding-top: 5px; padding-bottom: 5px">
        {text}
    </code>""".strip()


def horizontal_scrollable_div(html):
    """Wrap html with the necessary html to make horizontal scrolling possible.

    Examples
    ________

    display(HTML(utils.horizontal_scrollable_div(my_html)))

    Parameters
    ----------
    html: str
        Your HTML to wrap.

    Returns
    -------
    type
        Wrapped HTML.
    """

    return f"""
        <style>
            .mostly-customized-scrollbar {{
              display: block;
              width: 100%;
              overflow: auto;
            }}

            .mostly-customized-scrollbar::-webkit-scrollbar {{
              width: 5px;
              height: 8px;
              background-color: #aaa;
            }}

            .mostly-customized-scrollbar::-webkit-scrollbar-thumb {{
                background: #000;
                border-radius: 10px;
            }}
        </style>

        <div style="width=100%; display: flex; flex-wrap: nowrap; overflow-x: auto;">
            <div class="mostly-customized-scrollbar">
                {html}
            </div>
        </div>
    """


def is_notebook():
    """Returns true if the environment is a jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # pragma: no cover
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except ModuleNotFoundError or NameError:
        return False  # Probably standard Python interpreter


def is_test():  # pragma: no cover
    """
    Returns true if ADS is in test mode.
    """
    from ads import test_mode

    return test_mode


@deprecated(
    "2.6.8",
    details="Deprecated, use: ads.set_auth(auth='resource_principal')",
)
def is_resource_principal_mode():  # pragma: no cover
    """
    Returns true if ADS is in resource principal mode.
    """
    from ads import resource_principal_mode

    return resource_principal_mode


@deprecated(
    "2.6.8",
    details="Deprecated, use: from ads.common.auth import AuthState; AuthState().oci_config_path",
)
def oci_config_location():  # pragma: no cover
    """
    Returns oci configuration file location.
    """
    from ads.common.auth import AuthState

    return AuthState().oci_config_path


@deprecated(
    "2.6.8",
    details="Deprecated, use: from ads.common.auth import AuthState; AuthState().oci_key_profile",
)
def oci_key_profile():  # pragma: no cover
    """
    Returns key profile value specified in oci configuration file.
    """
    from ads.common.auth import AuthState

    return AuthState().oci_key_profile


def is_documentation_mode():  # pragma: no cover
    """
    Returns true if ADS is in documentation mode.
    """
    from ads import documentation_mode

    return documentation_mode


def is_debug_mode():  # pragma: no cover
    """
    Returns true if ADS is in debug mode.
    """
    from ads import debug_mode

    return debug_mode


@deprecated("2.3.1")
@runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
def print_user_message(
    msg, display_type="tip", see_also_links=None, title="Tip"
):  # pragma: no cover
    """This method is deprecated and will be removed in future releases.
    Prints in html formatted block one of tip|info|warn type.

    Parameters
    ----------
    msg : str or list
        The actual message to display.
        display_type is "module', msg can be a list of [module name, module package name], i.e. ["automl", "ads[ml]"]
    display_type : str (default 'tip')
        The type of user message.
    see_also_links : list of tuples in the form of [('display_name', 'url')]
    title : str (default 'tip')
        The title of user message.
    """

    if display_type.lower() == "error" and not is_documentation_mode():
        print("ERROR: {}".format(re.sub("<[^>]*>", "", msg)))

    if display_type.lower() == "module":
        if isinstance(msg, list()):
            module_name = msg[0]
            module_pkg = msg[1]
        else:
            module_name, module_pkg = msg, msg
        print(
            f"ERROR: {module_name} module not found. Make sure you have installed {module_pkg} in order to download all of necessary modules."
        )

    if is_documentation_mode() and is_notebook():
        if display_type.lower() == "tip":
            if "\n" in msg:
                t = "<b>{}:</b>".format(title.upper().strip()) if title else ""

                user_message = "{}{}".format(
                    t,
                    "".join(
                        [
                            "<br>&nbsp;&nbsp;+&nbsp;{}".format(x.strip())
                            for x in msg.strip().split("\n")
                        ]
                    ),
                )
            else:
                user_message = "{}".format(msg.strip().replace("\n", "<br>"))

            from IPython.core.display import HTML, display

            display(
                HTML(
                    f"""
                <div style="padding: 7px;
                            border-radius: 4px;
                            background-color: #d4ecd9;
                            margin_bottom: 5px;">
                    <p>{user_message}</p>
                </div>

                """
                )
            )

        elif display_type.lower() == "warning":
            user_message = "{}".format(msg.strip().replace("\n", "<br>"))

            display(
                HTML(
                    f"""
                <div style="padding: 7px;
                            border-radius: 4px;
                            background-color: #fcc5c5;
                            margin_bottom: 5px;">
                    <h3>Warning:</h3>
                    <p>{user_message}</p>
                </div>

                """
                )
            )

        elif display_type.lower() == "error":
            user_message = "{}".format(msg.strip().replace("\n", "<br>"))

            display(
                HTML(
                    f"""
                <div style="padding: 7px;
                            border-radius: 4px;
                            background-color: #4f053b;
                            color: white;
                            margin_bottom: 5px;">
                    <h2>Error:</h2>
                    <p>{user_message}</p>
                </div>

                """
                )
            )

        elif display_type.startswith("info"):
            user_message = msg.strip().replace("\n", "<br>")

            if see_also_links:
                see_also_html = f"""
                <ul>
                {'<li>'.join([f"<a src='{url}'>{display_name}</a>"
                              for (display_name, url) in see_also_links])}
                </ul>
                """
            else:
                see_also_html = ""

            if title:
                title_html = f"""<div style="padding: 5px;
                            color: #588864;
                            border_bottom: 1px solid grey;
                            margin_bottom: 5px;">
                    <h2>{title.upper()}</h2>
                </div>"""
            else:
                title_html = ""

            display(
                HTML(
                    f"""
            <br>

            <div style="width: calc(100% -20px);
                        border-left: 8px solid #588864;
                        margin: 10px, 0, 10px, 0px;
                        padding: 10px">

                {title_html}
                <p>{user_message}</p>
                {see_also_html}
            </div>

            """
                )
            )


# take a series which can be interpreted as a dict, index=key, this
# function sorts by the values and takes the top-n values, returning
# a new series
#
def truncate_series_top_n(series, n=24):
    """
    take a series which can be interpreted as a dict, index=key, this
    function sorts by the values and takes the top-n values, and returns
    a new series
    """
    return series.sort_values(ascending=False).head(n)


#
# take a sequence (<string>, list(<string>), tuple(<string>), pd.Series(<string>) and Ellipsis'ize them at position n
#
def ellipsis_strings(raw, n=24):
    """
    takes a sequence (<string>, list(<string>), tuple(<string>), pd.Series(<string>) and Ellipsis'ize them at position n
    """
    if isinstance(raw, pd.core.indexes.base.Index):
        sequence = raw.astype(str).to_list()

    if isinstance(raw, str):
        return ellipsis_strings([raw], n)[0]

    sequence = list(raw) if not isinstance(raw, list) else raw

    result = []
    for s in sequence:
        if len(str(s)) <= n:
            result.append(s)
        else:
            n2 = int(n) // 2 - 3
            n1 = n - n2 - 3
            result.append("{0}...{1}".format(s[:n1], s[-n2:]))

    return result


def first_not_none(itr):
    """
    Returns the first non-none result from an iterable,
    similar to any() but return value not true/false
    """
    for x in itr:
        if x:
            return x
    return None


#
# checks to see if object is the same class as cls
#
def is_same_class(obj, cls):
    """
    checks to see if object is the same class as cls
    """
    if isinstance(cls, (list, tuple)):
        return any([obj.__class__.__name__ == x.__name__ for x in cls])
    else:
        return obj.__class__.__name__ == cls.__name__


def replace_spaces(lst):
    """
    Replace all spaces with underscores for strings in the list.

    Requires that the list contains strings for each element.

    lst: list of strings
    """
    return [s.replace(" ", "_") for s in lst]


def get_progress_bar(
    max_progress: int, description: str = "Initializing", verbose: bool = False
) -> TqdmProgressBar:
    """Returns an instance of the TqdmProgressBar class.

    Parameters
    ----------
    max_progress: int
        The number of steps for the progressbar.
    description: (str, optional). Defaults to "Initializing".
        The first step description.
    verbose: (bool, optional). Defaults to `False`
        If the progress should show the debug information.

    Returns
    -------
    TqdmProgressBar
        An instance of the TqdmProgressBar.
    """
    return TqdmProgressBar(
        max_progress, description=description, verbose=verbose or is_debug_mode()
    )


class JsonConverter(json.JSONEncoder):
    def default(self, obj):
        """
        Converts an object to JSON based on its type

        Parameters
        ----------
        obj: Object
            An object which is being converted to Json, supported types are pandas Timestamp, series, dataframe, or categorical or numpy ndarrays.

        Returns
        -------
        Json: A json repersentation of the object.
        """
        if isinstance(obj, pd.Timestamp):
            return obj.__str__()
        if isinstance(obj, pd.Series):
            return obj.values
        if isinstance(obj, pd.Categorical):
            return obj.get_values()
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.to_json())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(
            obj, (np.float_, np.float16, np.float32, np.float64, np.double)
        ):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def split_data(X, y, random_state=random_state, test_size=test_size):
    """
    Splits data using Sklearn based on the input type of the data.

    Parameters
    ----------
    X: a Pandas Dataframe
        The data points.
    y: a Pandas Dataframe
        The labels.
    random_state: int
        A random state for reproducability.
    test_size: int
        The number of elements that should be included in the test dataset.
    """
    return train_test_split(
        X, y, train_size=1 - test_size, test_size=test_size, random_state=random_state
    )


@runtime_dependency(module="sqlalchemy", install_from=OptionalDependency.DATA)
def get_sqlalchemy_engine(connection_url, *args, **kwargs):
    """
    The SqlAlchemny docs say to use a single engine per connection_url, this class will take
    care of that.

    Parameters
    ----------

    connection_url: string
        The URL to connect to

    Returns
    -------
    engine: SqlAlchemny engine
        The engine from which SqlAlchemny commands can be ran on
    """
    global _engines
    if connection_url not in _engines:
        #
        # Note: pool_recycle=1 is used here because sqlalchemy is free to drop inactive
        # connections. This will make sure they are recycled and available when we
        # need them.
        #
        # DAR: note: use echo=True to log engine output
        _engines[connection_url] = sqlalchemy.create_engine(
            connection_url, pool_recycle=10, *args, **kwargs
        )

    return _engines[connection_url]


def inject_and_copy_kwargs(kwargs, **args):
    """Takes in a dictionary and returns a copy with the args injected

    Examples
    ________
    >>> foo(arg1, args, utils.inject_and_copy_kwargs(kwargs, arg3=12, arg4=42))

    Parameters
    ----------
    kwargs : dict
        The original `kwargs`.
    **args : type
        A series of arguments, foo=42, bar=12 etc

    Returns
    -------
    d: dict
        new dictionary object that you can use in place of kwargs

    """

    d = kwargs.copy()
    for k, v in args.items():
        if k not in kwargs:
            d[k] = v  # inject args iff not already found
    return d


def flatten(d, parent_key=""):
    """
    Flattens nested dictionaries to a single layer dictionary

    Parameters
    ----------
    d : dict
        The dictionary that needs to be flattened
    parent_key : str
        Keys in the dictionary that are nested

    Returns
    -------
    a_dict: dict
        a single layer dictionary
    """
    items = []
    for k, v in d.items():
        new_key = k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)


def wrap_lines(li, heading=""):
    """
    Wraps the elements of iterable into multi line string of fixed width
    """
    return heading + "\n" + fill(str(list(li)), width=30) if len(li) > 0 else ""


def get_base_modules(model):
    """
    Get the base modules from an ADS model
    """
    add_bases = []
    if hasattr(model, "est"):
        add_bases = get_base_modules(model.est)
    try:
        if hasattr(model, "steps") and isinstance(model.steps, list):
            [add_bases.extend(get_base_modules(step)) for _, step in model.steps]
    except:
        pass
    return (
        add_bases + list(type(model).__mro__)
        if hasattr(type(model), "__mro__")
        else add_bases
    )


def extract_lib_dependencies_from_model(model) -> dict:
    """
    Extract a dictionary of library dependencies for a model

    Parameters
    ----------
    model

    Returns
    -------
    Dict: A dictionary of library dependencies.
    """
    from pkg_resources import get_distribution

    module_versions = {}
    modules_to_include = set(
        mod.__module__.split(".")[0]
        for mod in get_base_modules(model)
        if hasattr(mod, "__module__")
    )
    for mod in modules_to_include:
        if mod not in module_ignore:
            try:
                mod_name = lib_translator.get(mod, mod)
                module_versions[mod_name] = get_distribution(mod_name).version
            except:
                pass
    return module_versions


def generate_requirement_file(
    requirements: dict, file_path: str, file_name: str = "requirements.txt"
):
    """
    Generate requirements file at file_path.

    Parameters
    ----------
    requirements : dict
        Key is the library name and value is the version
    file_path : str
        Directory to save requirements.txt
    file_name : str
        Opional parameter to specify the file name
    """

    with open(os.path.join(file_path, file_name), "w") as req_file:
        for lib in requirements:
            if requirements[lib]:
                req_file.write("{}=={}\n".format(lib, requirements[lib]))
            else:
                req_file.write("{}\n".format(lib))


def _get_feature_type_and_dtype(column):
    feature_type = "unknown"
    dtype = column.dtype
    if dtype.name in ["category", "object", "boolean"]:
        feature_type = "categorical"
    elif is_numeric_dtype(dtype):
        feature_type = "continuous"
    elif is_datetime64_dtype(dtype):
        feature_type = "datetime"
    return feature_type, dtype.name


def to_dataframe(
    data: Union[
        list,
        tuple,
        pd.Series,
        np.ndarray,
        pd.DataFrame,
    ]
):
    """
    Convert to pandas DataFrame.

    Parameters
    ----------
    data: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
        Convert data to pandas DataFrame.

    Returns
    _______
    pd.DataFrame
        pandas DataFrame.

    """
    if isinstance(data, np.ndarray) and len(data.shape) > DIMENSION:
        raise NotImplementedError(
            f"Cannot convert a numpy array with size {data.shape} to a pandas DataFrame."
        )
    if (
        isinstance(data, np.ndarray)
        or isinstance(data, list)
        or isinstance(data, tuple)
    ):
        return pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        return data.to_frame()
    elif isinstance(data, dict):
        try:
            return pd.DataFrame.from_dict(data)
        except:
            raise NotImplementedError(
                "Cannot convert this dictionary to a pandas DataFrame. \
                    Check the structure to ensure it is tabular."
            )
    elif isinstance(data, pd.DataFrame):
        return data
    elif _is_dask_dataframe(data):
        return data.compute()
    else:
        raise NotImplementedError(
            f"The data type `{type(data)}` is not supported. Convert it to a pandas DataFrame."
        )


def _is_dask_dataframe(ddf):
    """
    Will determine if the given arg is a dask dataframe.
    Returns False if dask is not installed.
    """
    try:
        import dask.dataframe as dd

        return isinstance(ddf, dd.DataFrame)
    except:
        return False


def _is_dask_series(ddf):
    """
    Will determine if the given arg is a dask dataframe.
    Returns False if dask is not installed.
    """
    try:
        import dask.dataframe as dd

        return isinstance(ddf, dd.Series)
    except:
        return False


def _log_missing_module(module, package):
    """
    Log message for missing module
    """
    logger.error(f"The {module} module was not found. Install {package}.")


def _log_multivalue_feature_column_error():
    logger.error(
        "A feature column has more than one value. Only a single value is allowed."
    )


def _log_plot_high_cardinality_warning(s, length):
    logger.warning(
        f"There are too many distinct values for {s} ({length:,}) to plot. Only the top {MAX_DISPLAY_VALUES}, by cardinality, will be used."
    )


def snake_to_camel(name: str, capitalized_first_token: Optional[bool] = False) -> str:
    """Converts the snake case string to the camel representation.

    Parameters
    ----------
    name: str
        The name to convert.
    capitalized_first_token: (bool, optional). Defaults to False.
        Wether the first token needs to be capitalized or not.

    Returns
    -------
    str: The name converted to the camel representation.
    """
    tokens = name.split("_")
    return (tokens[0].capitalize() if capitalized_first_token else tokens[0]) + "".join(
        x.capitalize() if not x.isupper() else x for x in tokens[1:]
    )


def camel_to_snake(name: str) -> str:
    """Converts the camel case string to the snake representation.

    Parameters
    ----------
    name: str
        The name to convert.

    Returns
    -------
    str: The name converted to the snake representation.
    """
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def is_data_too_wide(
    data: Union[
        list,
        tuple,
        pd.Series,
        np.ndarray,
        pd.DataFrame,
    ],
    max_col_num: int,
) -> bool:
    """
    Returns true if the data has too many columns.

    Parameters
    ----------

    data: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
        A sample of data that will be used to generate schema.
    max_col_num : int.
        The maximum column size of the data that allows to auto generate schema.
    """
    assert (
        max_col_num and isinstance(max_col_num, int) and max_col_num > 0
    ), "The parameter `max_col_num` must be a positive integer."

    data_type = type(data)
    if data_type == pd.Series or _is_dask_series(data):
        return False
    elif data_type == pd.DataFrame or _is_dask_dataframe(data):
        col_num = len(data.columns)
    elif (
        # check the column size in model_schema() after converting to pd.dataframe
        isinstance(data, np.ndarray)
        or isinstance(data, list)
        or isinstance(data, tuple)
    ):
        return False
    else:
        raise TypeError(f"The data type `{type(data)}` is not supported.")

    return col_num > max_col_num


def get_files(directory: str, auth: Optional[Dict] = None):
    """List out all the file names under this directory.

    Parameters
    ----------
    directory: str
        The directory to list out all the files from.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    List
        List of the files in the directory.
    """
    directory = directory.rstrip("/")
    path_scheme = urlparse(directory).scheme or "file"
    storage_options = auth or authutil.default_signer()
    model_ignore_path = os.path.join(directory, ".model-ignore")
    if is_path_exists(model_ignore_path, auth=auth):
        with fsspec.open(model_ignore_path, "r", **storage_options) as f:
            ignore_patterns = f.read().strip().split("\n")
    else:
        ignore_patterns = []
    file_names = []
    fs = fsspec.filesystem(path_scheme, **storage_options)
    for root, dirs, files in fs.walk(directory):
        for name in files:
            file_names.append(os.path.join(root, name))
        for name in dirs:
            file_names.append(os.path.join(root, name))

    # return all files in remote directory.
    if directory.startswith("oci://"):
        directory = directory.lstrip("oci://")

    for ignore in ignore_patterns:
        if not ignore.startswith("#") and ignore.strip() != "":
            matches = []
            for file_name in file_names:
                if ignore.endswith("/"):
                    ignore = ignore[:-1] + "*"
                if not re.search(fnmatch.translate("/%s" % ignore.strip()), file_name):
                    matches.append(file_name)
            file_names = matches
    return [matched_file[len(directory) + 1 :] for matched_file in file_names]


def download_from_web(url: str, to_path: str) -> None:
    """Downloads a single file from http/https/ftp.

    Parameters
    ----------
    url : str
        The URL of the source file.
    to_path : path-like object
        Local destination path.

    Returns
    -------
    None
        Nothing
    """
    url_response = request.urlopen(url)
    with contextlib.closing(url_response) as fp:
        with open(to_path, "wb") as out_file:
            block_size = DEFAULT_BUFFER_SIZE * 8
            while True:
                block = fp.read(block_size)
                if not block:
                    break
                out_file.write(block)


def copy_from_uri(
    uri: str,
    to_path: str,
    unpack: Optional[bool] = False,
    force_overwrite: Optional[bool] = False,
    auth: Optional[Dict] = None,
) -> None:
    """Copies file(s) to local path. Can be a folder, archived folder or a separate file.
    The source files can be located in a local folder or in OCI Object Storage.

    Parameters
    ----------
    uri: str
        The URI of the source file or directory, which can be local path or
        OCI object storage URI.
    to_path: str
        The local destination path.
        If this is a directory, the source files will be placed under it.
    unpack : (bool, optional). Defaults to False.
        Indicate if zip or tar.gz file specified by the uri should be unpacked.
        This option has no effect on other files.
    force_overwrite: (bool, optional). Defaults to False.
        Whether to overwrite existing files or not.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    None
        Nothing

    Raises
    ------
    ValueError
        If destination path is already exist and `force_overwrite` is set to False.
    """
    if os.path.exists(to_path):
        if not force_overwrite:
            raise ValueError(
                "The destination path already exists. "
                "Set `force_overwrite` to True if you wish to overwrite."
            )
        shutil.rmtree(to_path, ignore_errors=True)

    scheme = urlparse(uri).scheme
    auth = auth or authutil.default_signer()

    with tempfile.TemporaryDirectory() as temp_dir:
        if unpack and str(uri).lower().endswith((".zip", ".tar.gz", ".gztar")):
            unpack_path = to_path
            to_path = temp_dir
        else:
            unpack_path = None

        fs = fsspec.filesystem(scheme, **auth)

        if not (uri.endswith("/") or fs.isdir(uri)) and os.path.isdir(to_path):
            to_path = os.path.join(to_path, os.path.basename(str(uri).rstrip("/")))

        fs.get(uri, to_path, recursive=True)

        if unpack_path:
            shutil.unpack_archive(to_path, unpack_path)


def copy_file(
    uri_src: str,
    uri_dst: str,
    force_overwrite: Optional[bool] = False,
    auth: Optional[Dict] = None,
    chunk_size: Optional[int] = DEFAULT_BUFFER_SIZE,
    progressbar_description: Optional[str] = "Copying `{uri_src}` to `{uri_dst}`",
    ignore_if_src_not_exists: Optional[bool] = False,
) -> str:
    """
    Copies file from `uri_src` to `uri_dst`.
    If `uri_dst` specifies a directory, the file will be copied into `uri_dst`
    using the base filename from `uri_src`.
    Returns the path to the newly created file.

    Parameters
    ----------
    uri_src: str
        The URI of the source file, which can be local path or OCI object storage URI.
    uri_dst: str
        The URI of the destination file, which can be local path or OCI object storage URI.
    force_overwrite: (bool, optional). Defaults to False.
        Whether to overwrite existing files or not.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.
    chunk_size: (int, optional). Defaults to `DEFAULT_BUFFER_SIZE`
        How much data can be copied in one iteration.
    progressbar_description: (str, optional). Defaults to `"Copying `{uri_src}` to `{uri_dst}`"`.
        Prefix for the progressbar.

    Returns
    -------
    str
        The path to the newly created file.

    Raises
    ------
    FileExistsError
        If a destination file exists and `force_overwrite` set to `False`.
    """
    chunk_size = chunk_size or DEFAULT_BUFFER_SIZE

    if not os.path.basename(uri_dst):
        uri_dst = os.path.join(uri_dst, os.path.basename(uri_src))
    src_path_scheme = urlparse(uri_src).scheme or "file"

    auth = auth or {}
    if src_path_scheme.lower() == "oci" and not auth:
        auth = authutil.default_signer()

    src_file_system = fsspec.filesystem(src_path_scheme, **auth)

    if not fsspec.filesystem(src_path_scheme, **auth).exists(uri_src):
        if ignore_if_src_not_exists:
            return uri_dst
        raise FileNotFoundError(f"The `{uri_src}` not exists.")

    file_size = src_file_system.info(uri_src)["size"]
    if not force_overwrite:
        dst_path_scheme = urlparse(uri_dst).scheme or "file"
        if fsspec.filesystem(dst_path_scheme, **auth).exists(uri_dst):
            raise FileExistsError(
                f"The `{uri_dst}` exists. Please use a new file name or "
                "set force_overwrite to True if you wish to overwrite."
            )

    with fsspec.open(uri_dst, mode="wb", **auth) as fwrite:
        with fsspec.open(uri_src, mode="rb", encoding=None, **auth) as fread:
            with tqdm.wrapattr(
                fread,
                "read",
                desc=progressbar_description.format(uri_src=uri_src, uri_dst=uri_dst),
                total=file_size,
                position=0,
                leave=False,
                colour="blue",
                file=sys.stdout,
            ) as ffrom:
                while True:
                    chunk = ffrom.read(chunk_size)
                    if not chunk:
                        break
                    fwrite.write(chunk)

    return uri_dst


def remove_file(file_path: str, auth: Optional[Dict] = None) -> None:
    """
    Reoves file.

    Parameters
    ----------
    file_path: str
        The path of the source file, which can be local path or OCI object storage URI.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    None
        Nothing.
    """
    scheme = urlparse(file_path).scheme
    auth = auth or (scheme and authutil.default_signer()) or {}
    fs = fsspec.filesystem(scheme, **auth)
    try:
        fs.rm(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"`{file_path}` not found.")
    except Exception as e:
        raise e


def folder_size(path: str) -> int:
    """Recursively calculating a size of the `path` folder.

    Parameters
    ----------
    path: str
        Path to the folder.

    Returns
    -------
    int
        The size fo the folder in bytes.
    """
    if not path:
        return 0

    if os.path.isfile(path):
        return os.path.getsize(path)

    path = os.path.join(path.rstrip("/"), "**")
    return sum(
        os.path.getsize(f) for f in glob.glob(path, recursive=True) if os.path.isfile(f)
    )


def human_size(num_bytes: int, precision: Optional[int] = 2) -> str:
    """Converts bytes size to a string representing its value in B, KB, MB and GB.

    Parameters
    ----------
    num_bytes: int
        The size in bytes.
    precision: (int, optional). Defaults to 2.
        The precision of converting the bytes value.

    Returns
    -------
    str
        A string representing the size in B, KB, MB and GB.
    """
    if not num_bytes:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    size_index = int(math.floor(math.log(num_bytes, 1024)))
    result_size = round(num_bytes / math.pow(1024, size_index), precision)
    return f"{result_size}{size_name[size_index]}"


def get_value(obj, attr, default=None):
    """Gets a copy of the value from a nested dictionary of an object with nested attributes.

    Parameters
    ----------
    obj :
        An object or a dictionary
    attr :
        Attributes as a string seprated by dot(.)
    default :
        Default value to be returned if attribute is not found.

    Returns
    -------
    Any:
        A copy of the attribute value. For dict or list, a deepcopy will be returned.

    """
    keys = attr.split(".")
    val = default
    for key in keys:
        if hasattr(obj, key):
            val = getattr(obj, key)
        elif hasattr(obj, "get"):
            val = obj.get(key, default)
        else:
            return default
        obj = val
    return copy.deepcopy(val)


def _filter_fn(adjective: str, word: str) -> bool:
    """Used to filter list of adjectives phonetically

    Parameters
    ----------
    adjective: str
        adjective word
    word: str
        word to see if should be included in list of alliterations

    Returns
    -------
    bool:
        filter or not
    """
    if adjective.startswith("f"):
        return word.startswith("f") or word.startswith("ph")
    elif adjective.startswith("q"):
        return word.startswith("q") or word.startswith("k")
    else:
        return word.startswith(adjective[0])


def get_random_name_for_resource() -> str:
    """Returns randomly generated easy to remember name. It consists from 1 adjective and 1 animal word,
    tailed by UTC timestamp (joined with '-'). This is an ADS default resource name generated for
    models, jobs, jobruns, model deployments, pipelines.

    Returns
    -------
    str
        Randomly generated easy to remember name for oci resources - models, jobs, jobruns, model deployments, pipelines.
        Example: polite-panther-2022-08-17-21:15.46; strange-spider-2022-08-17-23:55.02
    """

    adjective = random.choice(adjectives)
    animal = random.choice(
        list(filter(lambda x: _filter_fn(adjective, x), animals)) or animals
    )

    return "-".join(
        (
            adjective,
            animal,
            datetime.utcnow().strftime("%Y-%m-%d-%H:%M.%S"),
        )
    )


def batch_convert_case(spec: dict, to_fmt: str) -> Dict:
    """
    Convert the case of a dictionary of spec from camel to snake or vice versa.

    Parameters
    ----------
    spec: dict
        dictionary of spec to convert
    to_fmt: str
        format to convert to, can be "camel" or "snake"

    Returns
    -------
    dict
        dictionary of converted spec
    """
    if not spec:
        return spec

    converted = {}
    if to_fmt == "camel":
        converter = snake_to_camel
    else:
        converter = camel_to_snake
    for k, v in spec.items():
        if k == "spec":
            converted[converter(k)] = batch_convert_case(v, to_fmt)
        else:
            converted[converter(k)] = v
    return converted


def extract_region(auth: Optional[Dict] = None) -> Union[str, None]:
    """Extracts region information from the environment variables and signer.

    Parameters
    ----------
    auth: Dict
        The ADS authentication config used to initialize the client.
        Contains keys - config, signer and client_kwargs.

    Returns
    -------
    Union[str, None]
        The region identifier. For example: `us-ashburn-1`.
        Returns `None` if region cannot be extracted.
    """
    auth = auth or authutil.default_signer()

    if auth.get("config", {}).get("region"):
        return auth["config"]["region"]

    if (
        auth.get("signer")
        and hasattr(auth["signer"], "region")
        and auth["signer"].region
    ):
        return auth["signer"].region

    try:
        return json.loads(config.OCI_REGION_METADATA)["regionIdentifier"]
    except:
        pass

    return None


def is_path_exists(uri: str, auth: Optional[Dict] = None) -> bool:
    """Check if the given path which can be local path or OCI object storage URI exists.

    Parameters
    ----------
    uri: str
        The URI of the target, which can be local path or OCI object storage URI.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    bool: return True if the path exists.
    """
    path_scheme = urlparse(uri).scheme or "file"
    storage_options = {}
    if path_scheme != "file":
        storage_options = auth or authutil.default_signer()
    if fsspec.filesystem(path_scheme, **storage_options).exists(uri):
        return True
    return False


def upload_to_os(
    src_uri: str,
    dst_uri: str,
    auth: dict = None,
    parallel_process_count: int = DEFAULT_PARALLEL_PROCESS_COUNT,
    progressbar_description: str = "Uploading `{src_uri}` to `{dst_uri}`.",
    force_overwrite: bool = False,
):
    """Utilizes `oci.object_storage.Uploadmanager` to upload file to Object Storage.

    Parameters
    ----------
    src_uri: str
        The path to the file to upload. This should be local path.
    dst_uri: str
        Object Storage path, eg. `oci://my-bucket@my-tenancy/prefix``.
    auth: (Dict, optional) Defaults to None.
        default_signer()
    parallel_process_count: (int, optional) Defaults to 3.
        The number of worker processes to use in parallel for uploading individual
        parts of a multipart upload.
    progressbar_description: (str, optional) Defaults to `"Uploading `{src_uri}` to `{dst_uri}`"`.
        Prefix for the progressbar.
    force_overwrite: (bool, optional). Defaults to False.
        Whether to overwrite existing files or not.

    Returns
    -------
    Response: oci.response.Response
        The response from multipart commit operation or the put operation.

    Raise
    -----
    ValueError
        When the given `dst_uri` is not a valid Object Storage path.
    FileNotFoundError
        When the given `src_uri` does not exist.
    RuntimeError
        When upload operation fails.
    """
    if not os.path.exists(src_uri):
        raise FileNotFoundError(f"The give src_uri: {src_uri} does not exist.")

    if not ObjectStorageDetails.is_oci_path(
        dst_uri
    ) or not ObjectStorageDetails.is_valid_uri(dst_uri):
        raise ValueError(
            f"The given dst_uri:{dst_uri} is not a valid Object Storage path."
        )

    auth = auth or authutil.default_signer()

    if not force_overwrite and is_path_exists(dst_uri):
        raise FileExistsError(
            f"The `{dst_uri}` exists. Please use a new file name or "
            "set force_overwrite to True if you wish to overwrite."
        )

    upload_manager = object_storage.UploadManager(
        object_storage_client=OCIClientFactory(**auth).object_storage,
        parallel_process_count=parallel_process_count,
        allow_multipart_uploads=True,
        allow_parallel_uploads=True,
    )

    file_size = os.path.getsize(src_uri)
    with open(src_uri, "rb") as fs:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            position=0,
            leave=False,
            file=sys.stdout,
            desc=progressbar_description,
        ) as pbar:

            def progress_callback(progress):
                pbar.update(progress)

            bucket_details = ObjectStorageDetails.from_path(dst_uri)
            response = upload_manager.upload_stream(
                namespace_name=bucket_details.namespace,
                bucket_name=bucket_details.bucket,
                object_name=bucket_details.filepath,
                stream_ref=fs,
                progress_callback=progress_callback,
            )

    if response.status == 200:
        print(f"{src_uri} has been successfully uploaded to {dst_uri}.")
    else:
        raise RuntimeError(
            f"Failed to upload {src_uri}. Response code is {response.status}"
        )

    return response


def get_console_link(
    resource: str,
    ocid: str,
    region: str,
) -> str:
    """
    This method returns the web console link for the given resource.
    Parameters
    ----------
    resource: str
        identify the type of OCI resource. {model, model-deployments, notebook-sessions, jobs} is supported.
    ocid: str
        OCID of the resource
    region: str
        The Region Identifier that the client should connect to.

    Returns
    -------
    console_link_url: str
        a valid link to the console for the given resource
    """
    console_link_url = (
        f"https://cloud.oracle.com/data-science/{resource}/{ocid}?region={region}"
    )
    return console_link_url


def get_log_links(
    region: str,
    log_group_id: str,
    compartment_id: str = None,
    log_id: str = None,
    source_id: str = None,
) -> str:
    """
    This method returns the web console link for the given log ids.

    Parameters
    ----------
    log_group_id: str, required
        OCID of the resource
    log_id: str, optional
        OCID of the resource
    region: str
        The Region Identifier that the client should connect to.
    compartment_id: str, optional
        The compartment OCID of the resource.
    source_id: str, optional
        The OCID of the resource.

    Returns
    -------
    console_link_url: str
        a valid link to the console for the given resource.
    """
    console_link_url = ""
    if log_group_id and log_id:
        # format: https://cloud.oracle.com/logging/search?searchQuery=search "<compartment>/<log_group>/<log>" | source='<source>' | sort by datetime desc&regions=<region>
        query_range = f'''search "{compartment_id}/{log_group_id}/{log_id}"'''
        query_source = f"source='{source_id}'"
        sort_condition = f"sort by datetime desc&regions={region}"
        search_query = (
            f"search?searchQuery={query_range} | {query_source} | {sort_condition}"
        )
        console_link_url = f"https://cloud.oracle.com/logging/{search_query}"
    elif log_group_id:
        console_link_url = f"https://cloud.oracle.com/logging/log-groups/{log_group_id}?region={region}"

    return console_link_url
