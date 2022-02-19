#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import matplotlib as mpl
from cycler import cycler
import collections
import json
from enum import Enum
from textwrap import fill
import re
import os
import random
import string
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_dtype

import numpy as np
import pandas as pd
from IPython import get_ipython

from sklearn.model_selection import train_test_split

from IPython.core.display import display, HTML

from ads.common import logger
from ads.common.deprecate import deprecated
from ads.dataset.progress import TqdmProgressBar, DummyProgressBar
from typing import Union

# For Model / Model Artifact libraries
lib_translator = {"sklearn": "scikit-learn"}
module_ignore = ["builtins", "ads", "automl", "mlx"]

# up-sample if length of dataframe is less than or equal to MAX_LEN_FOR_UP_SAMPLING
MAX_LEN_FOR_UP_SAMPLING = 5000

# down-sample if ratio of minority to majority class is greater than or equal to MIN_RATIO_FOR_DOWN_SAMPLING
MIN_RATIO_FOR_DOWN_SAMPLING = 1 / 20

# Maximum distinct values by cardinality will be used for plotting
MAX_DISPLAY_VALUES = 10

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

# declare custom exception class
class FileOverwriteError(Exception):
    pass


# get number of core count
def get_cpu_count():
    """
    Returns the number of CPUs available on this machine
    """
    from cpuinfo import get_cpu_info

    return get_cpu_info()["count"]


@deprecated(
    "2.5.2", details="This method is being deprecated in favor of `get_cpu_count`"
)
def get_compute_accelerator_ncores():
    return get_cpu_count()


def get_oci_config():
    """
    Returns the OCI config location, and the OCI config profile.
    """
    oci_config_location = os.environ.get(
        "OCI_CONFIG_LOCATION", f"{os.environ['HOME']}/.oci/config"
    )  # os.environ['HOME'] == home/datascience
    oci_config_profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    return oci_config_location, oci_config_profile


def oci_key_location():
    """
    Returns the OCI key location
    """
    return os.environ.get(
        "OCI_CONFIG_DIR", os.path.join(os.path.expanduser("~"), ".oci")
    )


def oci_config_file():
    """
    Returns the OCI config file location
    """
    return os.path.join(oci_key_location(), "config")


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


#
# checks to see if in notebook mode
#


def is_notebook():
    """
    Returns true if the environment is a jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # pragma: no cover
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


#
# checks to see if in test mode
#
#
# ***FOR TESTING PURPOSE ONLY***
#
def is_test():  # pragma: no cover
    """
    Returns true if ADS is in test mode.
    """
    from ads import test_mode

    return test_mode


def is_resource_principal_mode():  # pragma: no cover
    """
    Returns true if ADS is in resource principal mode.
    """
    from ads import resource_principal_mode

    return resource_principal_mode


def oci_key_profile():
    from ads import oci_key_profile

    return oci_key_profile


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


#
# returns the first non-none result from an iterable, similar to any() but return value not true/false
#
def first_not_none(itr):
    """
    returns the first non-none result from an iterable, similar to any() but return value not true/false
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


def get_progress_bar(max_progress, description="Initializing"):
    """this will return an instance of ProgressBar, sensitive to the runtime environment"""

    #
    # this will return either a DummyProgressBar (non-notebook) or TqdmProgressBar (notebook environement)
    #
    if is_notebook():  # pragma: no cover
        return TqdmProgressBar(
            max_progress, description=description, verbose=is_debug_mode()
        )
    else:
        return DummyProgressBar()


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
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
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

    from sqlalchemy import create_engine

    global _engines
    if connection_url not in _engines:
        #
        # Note: pool_recycle=1 is used here because sqlalchemy is free to drop inactive
        # connections. This will make sure they are recycled and available when we
        # need them.
        #
        # DAR: note: use echo=True to log engine output
        _engines[connection_url] = create_engine(
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
        if isinstance(v, collections.MutableMapping):
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


class InvalidObjectStoragePath(Exception):
    """Invalid Object Storage Path."""

    pass


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


def _snake_to_camel(name, capitalized_first_token=False):
    tokens = name.split("_")
    return (tokens[0].capitalize() if capitalized_first_token else tokens[0]) + "".join(
        x.capitalize() if not x.isupper() else x for x in tokens[1:]
    )


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
