#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import os
import shlex
import tempfile
import warnings
from getopt import gnu_getopt

from ads.common.decorator.runtime_dependency import (OptionalDependency,
                                                     runtime_dependency)
from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime, Job
from ads.jobs.utils import get_dataflow_config
from ads.opctl.constants import (ADS_DATAFLOW_CONFIG_FILE_NAME,
                                 DEFAULT_ADS_CONFIG_FOLDER)


def dataflow(line, cell=None):
    opts, args = gnu_getopt(
        shlex.split(line),
        "f:a:c:t:n:hwo",
        longopts=[
            "filename=",
            "archive=",
            "config=",
            "help=",
            "watch=",
            "log-type=",
            "num-lines=",
            "overwrite=",
        ],
    )
    opts_dict = {k: v for k, v in opts}
    if len(args) == 0 and ("-h" in opts_dict or "--help" in opts_dict):
        print(
            f"Run `dataflow run -h` or `dataflow log -h` to see options for subcommands."
        )
        return
    if args[0] not in ("run", "log"):
        print(
            f"`dataflow` expects subcommand `run` or `log`. Use `dataflow log` or `dataflow run` with options."
        )
        return

    if args[0] == "run":
        dataflow_run(opts_dict, args, cell)
    else:

        dataflow_log(opts_dict, args)


def dataflow_run(options, args, cell):
    if "-h" in options or "--help" in options:
        print(
            f"""
Usage: dataflow run [OPTIONS] -- [ARGS]
Options:
    -f, --filename, optional, filename to save the script to. default is "script.py".
    -a, --archive, optional, uri to archive zip.
    -c, --config, optional, configuration passed to dataflow. by default loads from {os.path.join(DEFAULT_ADS_CONFIG_FOLDER, ADS_DATAFLOW_CONFIG_FILE_NAME)}.
    -w, --watch, optional, a flag indicating that dataflow run will be watched after submission.
    -o, --overwrite, optional, overwrite file in object storage when uploading script or archive.zip.
    -h, show this help message.
Args:
    arguments to pass to script.
        """
        )
        return
    if "-c" not in options and "--config" not in options:
        dataflow_config = get_dataflow_config()
    else:
        if "-c" in options:
            dataflow_config = json.loads(options["-c"])
        if "--config" in options:
            dataflow_config = json.loads(options["--config"])
    if "-f" in options:
        script_name = options["-f"]
    elif "--file" in options:
        script_name = options["--file"]
    else:
        script_name = "script.py"
    if "-a" in options:
        archive_name = options["-a"]
    elif "--archive" in options:
        archive_name = options["--archive"]
    elif hasattr(dataflow_config, "archive_uri") and dataflow_config.archive_uri:
        archive_name = dataflow_config.archive_uri
    else:
        archive_name = None
    with tempfile.TemporaryDirectory() as td:
        file = os.path.join(td, script_name)
        with open(file, "w") as f:
            f.write(cell)
        rt_spec = {
            "scriptPathURI": file,
            "scriptBucket": dataflow_config.pop("script_bucket"),
        }
        if len(args) > 1:
            rt_spec["args"] = args[1:]
        if archive_name:
            rt_spec["archiveUri"] = archive_name
            rt_spec["archiveBucket"] = dataflow_config.pop("archive_bucket", None)
            if not archive_name.startswith("oci://") and not rt_spec["archiveBucket"]:
                raise ValueError(
                    "`archiveBucket` has to be set in the config if `archive` is a local path."
                )
        rt = DataFlowRuntime(rt_spec)
        infra = DataFlow(spec=dataflow_config)
        if "-o" in options or "--overwrite" in options:
            df = infra.create(rt, overwrite=True)
        else:
            df = infra.create(rt)
        print("DataFlow App ID", df.id)
        df_run = df.run()
        print("DataFlow Run ID", df_run.id)
        print("DataFlow Run Page", df_run.run_details_link)
        if "-w" in options or "--watch" in options:
            df_run.watch()
            print(df_run.logs.application.stdout.tail())


def dataflow_log(options, args):
    if "-h" in options or "--help" in options:
        print(
            f"""
Usage: dataflow log [OPTIONS] [RUN_ID]
Options:
    -t, --log-type, optional, should be one of application, driver, executor. default is "application".
    -n, --num-lines, optional, show last `n` lines of the log
    -h, show this help message.
        """
        )
        return
    log_type = (
        options.get("-t", None) or options.get("--log-type", None) or "application"
    )
    if log_type not in ("application", "driver", "executor"):
        print("Log type should be one of application, driver, executor.")
        return
    n = options.get("-n", None) or options.get("--num-lines", None)
    n = int(n) if (n and len(n) > 0) else None
    if len(args) < 2:
        raise ValueError("DataFlow Run ID must be provided.")
    df_run = DataFlowRun.from_ocid(args[1])
    if log_type == "application":
        print(df_run.logs.application.stdout.tail(n=n))
    if log_type == "driver":
        print(df_run.logs.driver.stdout.tail(n=n))
    if log_type == "executor":
        print(df_run.logs.executor.stdout.tail(n=n))


@runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
def load_ipython_extension(ipython):
    ipython.register_magic_function(dataflow, "line_cell")
