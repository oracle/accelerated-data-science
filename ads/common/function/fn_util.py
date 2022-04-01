#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import os
from jinja2 import Environment, PackageLoader

# FunctionAttributes = namedtuple('FunctionAttributes',['schema_version','name','version','runtime','entrypoint','memory'])
fn_config = None
env = Environment(loader=PackageLoader("ads", "templates"))


def prepare_fn_attributes(
    func_name: str,
    schema_version=20180708,
    version=None,
    python_runtime=None,
    entry_point=None,
    memory=None,
) -> dict:
    """
    Workaround for collections.namedtuples. The defaults are not supported.
    """
    function_attributes = {}
    function_attributes["schema_version"] = schema_version
    function_attributes["name"] = func_name
    function_attributes["version"] = (
        version if version else get_function_config()["fn_conf"]["version"]
    )
    function_attributes["runtime"] = (
        python_runtime
        if python_runtime
        else get_function_config()["fn_conf"]["runtime"]
    )
    function_attributes["entrypoint"] = (
        entry_point if entry_point else get_function_config()["fn_conf"]["entrypoint"]
    )
    function_attributes["memory"] = (
        memory if memory else get_function_config()["fn_conf"]["memory"]
    )

    return function_attributes


def get_function_config() -> dict:
    """
    Returns dictionary loaded from func_conf.yaml
    """
    global fn_config
    if fn_config:
        return fn_config
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "func_conf.yaml")
    ) as fn_config_file:
        func_config = yaml.load(fn_config_file, Loader=yaml.FullLoader)
    return func_config


def generate_fn_artifacts(
    path: str,
    fn_name: str = None,
    fn_attributes=None,
    artifact_type_generic=False,
    **kwargs,
):
    """
    Generates artifacts for fn (https://fnproject.io) at the provided path -
        * func.py
        * func.yaml
        * requirements.txt if not there. If exists appends fdk to the file.
        * score.py

    Parameters
    ----------
        path: str
            Target folder where the artifacts are placed.
        fn_attributes: dict
            dictionary specifying all the function attributes as described in https://github.com/fnproject/docs/blob/master/fn/develop/func-file.md
        artifact_type_generic: bool
            default is False. This attribute decides which template to pick for score.py. If True, it is assumed that the code to load is provided by the user.

    """
    assert fn_name or fn_attributes, (
        "Must provide either fn_name or fn_attributes. You may use "
        "ads.common.artifact.fn_util.prepare_fn_attributes for creating fn_attributes "
    )
    function_attributes = prepare_fn_attributes(fn_name) if fn_name else fn_attributes

    progress = kwargs.get("progress", None)
    if progress:
        progress.update("Writing func.yaml")
    with open(os.path.join(path, "func.yaml"), "w") as fnyaml_file:
        yaml.dump(function_attributes, fnyaml_file)

    if progress:
        progress.update("Writing func.py")
    func_template = env.get_template("func.jinja2")
    with open(os.path.join(path, "func.py"), "w") as func_fl:
        func_fl.write(func_template.render(score_module="score"))


def write_score(path, **kwargs):
    serializer = kwargs.get("serializer", "default")
    if serializer not in get_function_config()["models"]:
        serializer = "default"

    model_name = kwargs.get(
        "model_name", get_function_config()["models"][serializer]["file_name"]
    )
    transformer_name = (
        get_function_config()["models"][serializer]["transformer_name"]
        if serializer in get_function_config()["models"]
        else "onnx_data_transformer.json"
    )
    misc_imports = (
        kwargs.get("misc_imports", list())
        + get_function_config()["models"][serializer]["misc_imports"]
    )
    underlying_model = kwargs.get("underlying_model", "NOTFOUND")
    sklearn_model = underlying_model in ["sklearn", "xgboost"]

    if underlying_model == "automl":
        jinja_template_filename = "score_oracle_automl"
    else:
        jinja_template_filename = kwargs.get(
            "input_file",
            get_function_config()["models"][serializer].get("input_file", "score"),
        )
    scorefn_template = env.get_template(f"{jinja_template_filename}.jinja2")
    with open(os.path.join(path, "score.py"), "w") as sfl:
        sfl.write(
            scorefn_template.render(
                model_file_name=model_name,
                transfromer_file_name=transformer_name,
                misc_imports=misc_imports,
                sklearn_model=sklearn_model,
                underlying_model_type=kwargs.get("_underlying_model", None),
            )
        )
