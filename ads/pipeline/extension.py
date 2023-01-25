#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from getopt import gnu_getopt
import shlex
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.pipeline import Pipeline, PipelineRun, LogType

MAXIMUM_TIMEOUT_SECONDS = 1800
LOG_RECORDS_LIMIT = 100


def pipeline(line, cell=None):
    opts, args = gnu_getopt(
        shlex.split(line),
        "f:o:l:m:n:wxjpstdh",
        longopts=[
            "file=",
            "ocid=",
            "log-type=",
            "max-wait-seconds=",
            "number=",
            "watch",
            "text",
            "no-delete-related-job-runs",
            "no-delete-related-pipeline-runs",
            "succeed-on-not-found",
            "tail",
            "head",
        ],
    )

    PIPELINE_MAGIC_COMMANDS = {
        "run": pipeline_run,
        "log": pipeline_log,
        "cancel": pipeline_cancel,
        "delete": pipeline_delete,
        "show": pipeline_show,
        "status": pipeline_status,
    }

    opts_dict = {k: v for k, v in opts}
    if len(args) == 0 and "-h" in opts_dict:
        print(
            f"""
Usage: pipeline [SUBCOMMAND]
Subcommand:
    run, run a pipeline from YAML or an existing ocid.
    log, stream the logs from pipeline run.
    cancel, cancel a pipeline run.
    delete, delete pipeline or pipeline run.
    show, show the pipeline orchestration.
    status, show the real-time status of a pipeline run.

Run pipeline [SUBCOMMAND] -h to see more details.
        """
        )
        return
    if args[0] not in PIPELINE_MAGIC_COMMANDS:
        commands = ", ".join(list(PIPELINE_MAGIC_COMMANDS.keys()))
        print(f"`pipeline` expects subcommand {commands}.")
        return

    PIPELINE_MAGIC_COMMANDS[args[0]](opts_dict, args)


def pipeline_run(options, args):
    if "-h" in options:
        print(
            f"""
Usage: pipeline run [OPTIONS]
Options:
    -f, --file, optional, uri to the YAML.
    -o, --ocid, optional, ocid of existing pipeline.
    -w, --watch, optional, a flag indicating that pipeline run will be watched after submission.
    -l, --log-type, optional, should be either custom_log or service_log. default is "custom_log".
    -h, show this help message.
        """
        )
        return
    file = None
    ocid = None
    if "-f" in options:
        file = options["-f"]
    elif "--file" in options:
        file = options["--file"]
    if "-o" in options:
        ocid = options["-o"]
    elif "--ocid" in options:
        ocid = options["--ocid"]

    if ocid:
        pipeline = Pipeline.from_id(ocid)
    elif file:
        pipeline = Pipeline.from_yaml(uri=file)
        pipeline.create()
    else:
        print(
            f"`pipeline run` expects YAML file uri or pipeline ocid. Use `pipeline run` with options."
        )
        return

    print("Pipeline OCID:", pipeline.id)
    pipeline_run = pipeline.run()
    print("Pipeline Run OCID:", pipeline_run.id)
    print(pipeline)

    if "-w" in options or "--watch" in options:
        log_type = (
            options.get("-l", None)
            or options.get("--log-type", None)
            or LogType.CUSTOM_LOG
        )
        if log_type not in (LogType.CUSTOM_LOG, LogType.SERVICE_LOG):
            print(
                "Log type should be either custom_log or service_log. Using default custom_log."
            )
            log_type = LogType.CUSTOM_LOG
        print("Streaming logs...")
        pipeline_run.watch(log_type=log_type)


def pipeline_log(options, args):
    if "-h" in options:
        print(
            f"""
Usage: pipeline log [OPTIONS] [RUN_ID]
Options:
    -l, --log-type, optional, should be either custom_log, service_log or None. default is None.
    -t, --tail, a flag to show the most recent log records.
    -d, --head, a flag to show the preceding log records.
    -n, --number, number of lines of logs to be printed. Defaults to 100.
    -h, show this help message.
        """
        )
        return
    if len(args) < 2:
        raise ValueError("Pipeline Run ID must be provided.")
    log_type = options.get("-l", None) or options.get("--log-type", None)
    if log_type and log_type not in (LogType.CUSTOM_LOG, LogType.SERVICE_LOG):
        print("Log type should be either custom_log, service_log or None.")
        return

    limit = LOG_RECORDS_LIMIT
    if "-n" in options:
        limit = int(options["-n"])
    elif "--number" in options:
        limit = int(options["--number"])

    pipeline_run = PipelineRun.from_ocid(args[1])
    if "-t" in options or "--tail" in options:
        pipeline_run.logs(log_type=log_type).tail(limit=limit)
    elif "-d" in options or "--head" in options:
        pipeline_run.logs(log_type=log_type).head(limit=limit)
    else:
        pipeline_run.watch(log_type=log_type)


def pipeline_cancel(options, args):
    if "-h" in options:
        print(
            f"""
Usage: pipeline cancel [RUN_ID]
Options:
    -h, show this help message.
        """
        )
        return
    if len(args) < 2:
        raise ValueError("Pipeline Run ID must be provided.")

    pipeline_run = PipelineRun.from_ocid(args[1])
    pipeline_run.cancel()
    print(f"Pipeline Run {args[1]} has been cancelled.")


def pipeline_delete(options, args):
    if "-h" in options:
        print(
            f"""
Usage: pipeline delete [OCID]
Options:
    -j, --no-delete-related-job-runs, a flag to not delete the related job runs.
    -p, --no-delete-related-pipeline-runs, a flag to not delete related pipeline runs.
    -m, --max-wait-seconds, integer, maximum wait time in second for delete to complete. Defaults to 1800.
    -s, --succeeded-on-not-found, to flag to return successfully if the data we're waiting on is not found.
    -h, show this help message.
        """
        )
        return
    if len(args) < 2:
        raise ValueError("Pipeline or Pipeline Run ID must be provided.")

    delete_related_job_runs = True
    delete_related_pipeline_runs = True
    max_wait_seconds = MAXIMUM_TIMEOUT_SECONDS
    succeed_on_not_found = False

    if "-j" in options or "--no-delete-related-job-runs" in options:
        delete_related_job_runs = False

    if "-p" in options or "--no-delete-related-pipeline-runs" in options:
        delete_related_pipeline_runs = False

    if "-m" in options:
        max_wait_seconds = int(options["-m"])
    elif "--max-wait-seconds" in options:
        max_wait_seconds = int(options["--max-wait-seconds"])

    if "-s" in options or "--succeeded-on-not-found" in options:
        succeed_on_not_found = True

    id = args[1]
    if "datasciencepipelinerun" in id:
        pipeline_run = PipelineRun.from_ocid(id)
        pipeline_run.delete(
            delete_related_job_runs=delete_related_job_runs,
            max_wait_seconds=max_wait_seconds,
        )
        print(f"Pipeline Run {args[1]} has been deleted.")
    elif "datasciencepipeline" in id:
        pipeline = Pipeline.from_ocid(id)
        pipeline.delete(
            delete_related_pipeline_runs=delete_related_pipeline_runs,
            delete_related_job_runs=delete_related_job_runs,
            max_wait_seconds=max_wait_seconds,
            succeed_on_not_found=succeed_on_not_found,
        )
        print(f"Pipeline {args[1]} has been deleted.")
    else:
        print(f"`pipeline delete` expects a valid pipeline or pipeline run id.")


def pipeline_show(options, args):
    if "-h" in options:
        print(
            f"""
Usage: pipeline show [OCID]
Options:
    -h, show this help message.
        """
        )
        return
    if len(args) < 2:
        raise ValueError("Pipeline ID must be provided.")

    pipeline = Pipeline.from_ocid(args[1])
    pipeline.show()


def pipeline_status(options, args):
    if "-h" in options:
        print(
            f"""
Usage: pipeline status [OPTIONS] [RUN_ID]
Options:
    -x, --text, optional, a flag to show the status in text format.
    -w, --watch, optional, a flag to wait until the completion of the pipeline run.
    If set, the rendered graph will be updated until the completion of the pipeline run,
    otherwise will render one graph with the current status.
    -h, show this help message.
        """
        )
        return
    if len(args) < 2:
        raise ValueError("Pipeline Run ID must be provided.")

    pipeline_run = PipelineRun.from_ocid(args[1])

    if "-x" in options or "--text" in options:
        if "-w" in options or "--watch" in options:
            pipeline_run.show(mode="text")
        else:
            pipeline_run.show(mode="text")
    else:
        if "-w" in options or "--watch" in options:
            pipeline_run.show(wait=True)
        else:
            pipeline_run.show()


@runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
def load_ipython_extension(ipython):
    ipython.register_magic_function(pipeline)
