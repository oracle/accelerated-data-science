#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy
import logging
import time
from typing import List, Optional

import oci
import yaml
from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.common.extended_enum import ExtendedEnumMeta
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.oci_logging import ConsolidatedLog, OCILog
from ads.jobs.builders.infrastructure.base import RunInstance
from ads.pipeline.ads_pipeline_step import PipelineStep
from ads.pipeline.visualizer.base import (
    GraphOrientation,
    PipelineVisualizer,
    StepStatus,
)
from ads.pipeline.visualizer.graph_renderer import PipelineGraphRenderer
from ads.pipeline.visualizer.text_renderer import PipelineTextRenderer

PIPELINE_RUN_TERMINAL_STATE = {
    StepStatus.FAILED,
    StepStatus.SUCCEEDED,
    StepStatus.CANCELED,
    StepStatus.DELETED,
    StepStatus.SKIPPED,
}

LOG_INTERVAL = 3
SLEEP_INTERVAL = 3
MAXIMUM_TIMEOUT_SECONDS = 1800
LOG_RECORDS_LIMIT = 100
ALLOWED_OPERATION_KWARGS = [
    "allow_control_chars",
    "retry_strategy",
    "delete_related_job_runs",
    "if_match",
    "opc_request_id",
]
ALLOWED_WAITER_KWARGS = [
    "max_interval_seconds",
    "max_wait_seconds",
    "wait_callback",
    "fetch_func",
]

logger = logging.getLogger(__name__)


class LogType(str, metaclass=ExtendedEnumMeta):
    CUSTOM_LOG = "custom_log"
    SERVICE_LOG = "service_log"


class ShowMode(str, metaclass=ExtendedEnumMeta):
    GRAPH = "graph"
    TEXT = "text"


class StepType(str, metaclass=ExtendedEnumMeta):
    ML_JOB = "ML_JOB"
    CUSTOM_SCRIPT = "CUSTOM_SCRIPT"


class LogNotConfiguredError(Exception):   # pragma: no cover
    pass


class PipelineRun(
    OCIDataScienceMixin, oci.data_science.models.PipelineRun, RunInstance
):
    """
    Attributes
    ----------
    pipeline: Pipeline
        Returns the ADS pipeline object for run instance.
    status: str
        Returns Lifecycle status.
    custom_logging: OCILog
        Returns the OCILog object containing the custom logs from the pipeline.

    Methods
    -------
    create(self) -> PipelineRun
        Creates an OCI pipeline run.
    delete(self, delete_related_job_runs: Optional[bool] = True, max_wait_seconds: Optional[int] = MAXIMUM_TIMEOUT, **kwargs) -> PipelineRun
        Deletes an OCI pipeline run.
    cancel(self, maximum_timeout: int = MAXIMUM_TIMEOUT) -> PipelineRun
        Cancels an OCI pipeline run.
    watch(self, steps: List[str] = None, interval: float = LOG_INTERVAL, log_type: str = LogType.CUSTOM_LOG, *args) -> PipelineRun
        Watches the pipeline run until it finishes.
    list(cls, pipeline_id: str, compartment_id: Optional[str] = None, **kwargs) -> List[PipelineRun]:
        Lists pipeline runs for a given pipeline.
    to_yaml(self) -> str
        Serializes the object into YAML string.
    show(self, mode: str = ShowMode.GRAPH, wait: bool = False, rankdir: str = GraphOrientation.TOP_BOTTOM) -> None
        Renders pipeline run. Can be `text` or `graph` representation.
    to_svg(self, uri: str = None, rankdir: str = GraphOrientation.TOP_BOTTOM, **kwargs)
        Renders pipeline run graph to SVG.
    sync(self) -> None
        Syncs status of Pipeline run.
    """

    _DETAILS_LINK = (
        "https://console.{region}.oraclecloud.com/data-science/pipeline-runs/{id}"
    )

    def __init__(
        self,
        config: dict = None,
        signer: oci.signer.Signer = None,
        client_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(config, signer, client_kwargs, **kwargs)
        self._service_logging = None
        self._custom_logging = None
        self._pipeline = None

        self._graphViz = PipelineVisualizer().with_renderer(
            PipelineGraphRenderer(show_status=True)
        )
        self._textViz = PipelineVisualizer().with_renderer(PipelineTextRenderer())

    def sync(self, **kwargs) -> None:
        """Syncs status of the Pipeline Run.

        Returns
        -------
        None
        """
        super().sync(**kwargs)
        self._graphViz.with_pipeline(self.pipeline).with_pipeline_run(self)
        self._textViz.with_pipeline(self.pipeline).with_pipeline_run(self)
        return self

    def show(
        self,
        mode: str = ShowMode.GRAPH,
        wait: bool = False,
        rankdir: str = GraphOrientation.TOP_BOTTOM,
    ) -> None:
        """
        Renders pipeline run. Can be `text` or `graph` representation.

        Parameters
        ----------
        mode: (str, optional). Defaults to `graph`.
            Pipeline run display mode. Allowed values: `graph` or `text`.
        wait: (bool, optional). Default to `False`.
            Whether to wait until the completion of the pipeline run.
        rankdir: (str, optional). Default to `TB`.
            Direction of the rendered graph. Allowed Values: `TB` or `LR`.
            Applicable only for graph mode.

        Returns
        -------
        None
        """
        self.sync()
        renderer = self._graphViz if mode.lower() == ShowMode.GRAPH else self._textViz
        if not wait:
            renderer.render(rankdir=rankdir)
            return
        self._show(renderer, rankdir=rankdir)

    def to_svg(
        self, uri: str = None, rankdir: str = GraphOrientation.TOP_BOTTOM, **kwargs
    ) -> str:
        """
        Renders pipeline run graph to SVG.

        Parameters
        ----------
        uri: (string, optional). Defaults to `None`.
            URI location to save the SVG string.
        rankdir: (str, optional). Default to `TB`.
            Direction of the rendered graph. Allowed Values: `TB` or `LR`.
            Applicable only for graph mode.

        Returns
        -------
        str
            Pipeline run graph in svg format.
        """
        self.sync()
        return self._graphViz.to_svg(uri=uri, rankdir=rankdir, **kwargs)

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def _show(
        self,
        viz,
        rankdir: str = GraphOrientation.TOP_BOTTOM,
        refresh_interval=SLEEP_INTERVAL,
    ):
        """
        Renders pipeline run in text or graph until the completion of the pipeline.

        Parameters
        ----------
        viz: PipelineRenderer
            The `PipelineTextRenderer` or `PipelineGraphRenderer` object.
        rankdir: (str, optional). Default to `TB`.
            Direction of the rendered graph. Allowed Values: `TB` or `LR`.
            Applicable only for graph mode.
        refresh_interval: (int, optional). Defaults to 5.
            Time interval in seconds to refresh pipeline run status.

        Returns
        -------
        None
        """
        from IPython.display import clear_output

        try:
            while self.status not in PIPELINE_RUN_TERMINAL_STATE:
                clear_output(wait=True)
                viz.render(rankdir=rankdir)
                time.sleep(refresh_interval)

            clear_output(wait=True)
            viz.render(rankdir=rankdir)
        except KeyboardInterrupt:
            pass

    def logs(self, log_type: str = None) -> ConsolidatedLog:
        """Builds the consolidated log for pipeline run.

        Parameters
        ----------
        log_type: str
            The log type of the pipeline run. Defaults to None.
            Can be custom_log, service_log or None.

        Returns
        -------
        ConsolidatedLog
            The ConsolidatedLog instance.
        """
        logging_list = []
        if not log_type:
            try:
                logging_list.append(self.custom_logging)
            except LogNotConfiguredError:
                pass

            try:
                logging_list.append(self.service_logging)
            except LogNotConfiguredError:
                pass

            if not logging_list:
                raise LogNotConfiguredError(
                    "Neither `custom` nor `service` log was configured for the pipeline run."
                )
        elif log_type == LogType.SERVICE_LOG:
            logging_list = [self.service_logging]
        elif log_type == LogType.CUSTOM_LOG:
            logging_list = [self.custom_logging]
        else:
            raise ValueError(
                "Parameter log_type should be either custom_log, service_log or None."
            )

        return ConsolidatedLog(*logging_list)

    @property
    def pipeline(self):
        """Returns the ADS Pipeline instance.
        Step details will be synched with the Pipeline Run.

        Parameters
        ----------
        None

        Returns
        -------
        Pipeline
            The ADS Pipeline instance, where Step details will be synched with the Pipeline Run.
        """
        from ads.pipeline.ads_pipeline import Pipeline

        if not self._pipeline:
            self._pipeline = Pipeline.from_ocid(self.pipeline_id)
            self._sync_step_details()
        return self._pipeline

    @property
    def status(self) -> str:
        """Lifecycle status.

        Returns
        -------
        str
            Status in a string.
        """
        self.sync()
        return self.lifecycle_state

    @property
    def custom_logging(self) -> OCILog:
        """The OCILog object containing the custom logs from the pipeline run."""
        if not self._custom_logging:
            self._check_log_details()

            while not self._stop_condition():
                # Break if pipeline run has log ID.
                if self.log_details.log_id:
                    break
                time.sleep(LOG_INTERVAL)

            self._custom_logging = OCILog(
                id=self.log_details.log_id,
                log_group_id=self.log_details.log_group_id,
                compartment_id=self.compartment_id,
                annotation="custom",
            )
        return self._custom_logging

    @property
    def service_logging(self) -> OCILog:
        """The OCILog object containing the service logs from the pipeline run."""
        if not self._service_logging:
            self._check_log_details()
            self._service_logging = self._get_service_logging()
        return self._service_logging

    def _check_log_details(self):
        if not self.log_details:
            raise LogNotConfiguredError(
                "Pipeline log is not configured. Make sure log group id is added."
            )
        if not self.log_details.log_group_id:
            raise LogNotConfiguredError(
                "Log group OCID is not specified for this pipeline. Call with_log_group_id to add it."
            )

    def _get_service_logging(self) -> OCILog:
        """Builds the OCI service log instance for pipeline run.

        Returns
        -------
        OCILog
            The OCILog instance.
        """
        log_summary = self._search_service_logs()

        if not log_summary:
            raise LogNotConfiguredError("Service log is not configured for pipeline.")

        # each pipeline can only have one service log
        service_log_id = log_summary[0].id
        return OCILog(
            id=service_log_id,
            log_group_id=self.log_details.log_group_id,
            compartment_id=self.compartment_id,
            annotation="service",
        )

    def _search_service_logs(self) -> List[oci.logging.models.log_summary.LogSummary]:
        """Search the service log of pipeline run based on
        log_group_id, source_service, source_resource and log_type.

        Returns
        -------
        list
            A list of oci.logging.models.log_summary.LogSummary.
        """
        return (
            OCILog(compartment_id=self.compartment_id)
            .client.list_logs(
                log_group_id=self.log_details.log_group_id,
                source_service=self.pipeline.CONST_SERVICE,
                source_resource=self.pipeline.id,
                log_type="SERVICE",
            )
            .data
        )

    def _sync_step_details(self) -> None:
        """Combines pipeline step details with override step details.

        Returns
        -------
        None
        """
        if not self._pipeline or not self._pipeline.step_details:
            return None

        updated_step_details = []
        for step in self._pipeline.step_details:
            updated_step_detail = copy.deepcopy(step.to_dict())
            # restore dependencies information
            updated_step_detail["spec"]["dependsOn"] = step.depends_on
            # override step details if necessary
            if self.step_override_details:
                for override_step in self.step_override_details:
                    if type(override_step) == dict:
                        break
                    if step.name == override_step.step_name:
                        if override_step.step_configuration_details:
                            if (
                                "stepConfigurationDetails"
                                not in updated_step_detail["spec"]
                            ):
                                updated_step_detail["spec"][
                                    "stepConfigurationDetails"
                                ] = {}

                            if (
                                override_step.step_configuration_details.maximum_runtime_in_minutes
                            ):
                                updated_step_detail["spec"]["stepConfigurationDetails"][
                                    "maximumRuntimeInMinutes"
                                ] = (
                                    override_step.step_configuration_details.maximum_runtime_in_minutes
                                )
                            if (
                                override_step.step_configuration_details.environment_variables
                            ):
                                updated_step_detail["spec"]["stepConfigurationDetails"][
                                    "environmentVariables"
                                ] = (
                                    override_step.step_configuration_details.environment_variables
                                )
                            if (
                                override_step.step_configuration_details.command_line_arguments
                            ):
                                updated_step_detail["spec"]["stepConfigurationDetails"][
                                    "commandLineArguments"
                                ] = (
                                    override_step.step_configuration_details.command_line_arguments
                                )

            updated_step_details.append(PipelineStep.from_dict(updated_step_detail))
        self._pipeline.with_step_details(updated_step_details)

    def _set_service_logging_resource(self, service_logging: OCILog):
        """Sets the service logging resource for pipeline run.

        Parameters
        ----------
        service_logging: OCILog instance.
            The OCILog instance.
        """
        self._service_logging = service_logging

    def create(self) -> "PipelineRun":
        """Creates an OCI pipeline run.

        Returns
        -------
        PipelineRun:
            Pipeline run instance (self).
        """
        self.load_properties_from_env()
        response = self.client.create_pipeline_run(
            self.to_oci_model(oci.data_science.models.CreatePipelineRunDetails)
        )
        self.update_from_oci_model(response.data)
        return self

    def delete(
        self,
        delete_related_job_runs: Optional[bool] = True,
        max_wait_seconds: Optional[int] = MAXIMUM_TIMEOUT_SECONDS,
        **kwargs,
    ) -> "PipelineRun":
        """Deletes an OCI pipeline run.

        Parameters
        ----------
        delete_related_job_runs: bool, optional
            Specify whether to delete related JobRuns or not. Defaults to True.
        max_wait_seconds: int, optional
            The maximum time to wait, in seconds. Defaults to 1800.

        kwargs: optional
        The kwargs to be executed when deleting the pipeline.
        The allowed keys are:
        * "allow_control_chars": bool, to indicate whether or not this request should
        allow control characters in the response object. By default, the response will
        not allow control characters in strings.
        * "retry_strategy": obj, to apply to this specific operation/call. This will
        override any retry strategy set at the client-level. This should be one of the
        strategies available in the :py:mod:`~oci.retry` module. This operation will not
        retry by default, users can also use the convenient :py:data:`~oci.retry.DEFAULT_RETRY_STRATEGY`
        provided by the SDK to enable retries for it. The specifics of the default retry
        strategy are described `here <https://docs.oracle.com/en-us/iaas/tools/python/latest/sdk_behaviors/retries.html>`__.
        To have this operation explicitly not perform any retries, pass an instance of :py:class:`~oci.retry.NoneRetryStrategy`.
        * "if_match": str, for optimistic concurrency control. In the PUT or DELETE call
        for a resource, set the `if-match` parameter to the value of the etag from a
        previous GET or POST response for that resource. The resource is updated or
        deleted only if the `etag` you provide matches the resource's current `etag` value.
        * "opc_request_id": str, unique Oracle assigned identifier for the request.
        If you need to contact Oracle about a particular request, then provide the request ID.
        * "max_interval_seconds": int, the maximum interval between queries, in seconds.
        * "wait_callback": A function which will be called each time that we have to do an initial
        wait (i.e. because the property of the resource was not in the correct state,
        or the ``evaluate_response`` function returned False). This function should take two
        arguments - the first argument is the number of times we have checked the resource,
        and the second argument is the result of the most recent check.
        * "fetch_func": A function to be called to fetch the updated state from the server.
        This can be used if the call to check for state needs to be more complex than a single
        GET request. For example, if the goal is to wait until an item appears in a list,
        fetch_func can be a function that paginates through a full list on the server.

        Returns
        -------
        PipelineRun:
            Pipeline run instance (self).
        """
        operation_kwargs = {"delete_related_job_runs": delete_related_job_runs}
        waiter_kwargs = {"max_wait_seconds": max_wait_seconds}
        for key, value in kwargs.items():
            if key in ALLOWED_OPERATION_KWARGS:
                operation_kwargs[key] = value
            elif key in ALLOWED_WAITER_KWARGS:
                waiter_kwargs[key] = value

        self.client_composite.delete_pipeline_run_and_wait_for_state(
            pipeline_run_id=self.id,
            wait_for_states=[PipelineRun.LIFECYCLE_STATE_DELETED],
            operation_kwargs=operation_kwargs,
            waiter_kwargs=waiter_kwargs,
        )
        return self.sync()

    def cancel(self, maximum_timeout: int = MAXIMUM_TIMEOUT_SECONDS) -> "PipelineRun":
        """Cancels an OCI pipeline run.

        Parameters
        ----------
        maximum_timeout: int, optional
            The maximum timeout to cancel the pipeline run. Defaults to 1800 seconds.

        Returns
        -------
        PipelineRun:
            Pipeline run instance (self).
        """
        self.client.cancel_pipeline_run(self.id)

        time_counter = 0
        while self.sync().lifecycle_state not in [
            PipelineRun.LIFECYCLE_STATE_CANCELED,
            PipelineRun.LIFECYCLE_STATE_FAILED,
        ]:
            time.sleep(SLEEP_INTERVAL)
            if time_counter > maximum_timeout:
                print(
                    "Pipeline run stopping after ",
                    maximum_timeout,
                    " seconds of not reaching CANCELLED state.",
                )
                break
            time_counter += SLEEP_INTERVAL

        if self.sync().lifecycle_state != PipelineRun.LIFECYCLE_STATE_CANCELED:
            raise Exception("Error occurred in attempt to cancel the pipeline run.")
        return self

    def watch(
        self,
        steps: List[str] = None,
        interval: float = LOG_INTERVAL,
        log_type: str = None,
        *args,
    ) -> "PipelineRun":
        """Watches the pipeline run until it finishes.
        This method will keep streamming the service log of the pipeline run until it's succeeded, failed or cancelled.

        Parameters
        ----------
        steps: (List[str], optional). Defaults to None.
            Pipeline steps passed in to filter the logs.
        interval: (float, optional). Defaults to 3 seconds.
            Time interval in seconds between each request to update the logs.
        log_type: (str, optional). Defaults to None.
            The log type. Can be `custom_log`, `service_log` or None.
        *args:
            Pipeline steps passed in to filter the logs.
            Example: `.watch("step1", "step2")`

        Examples
        --------
        >>> .watch()
        >>> .watch(log_type="service_log")
        >>> .watch("step1", "step2", log_type="custom_log", interval=3)
        >>> .watch(steps=["step1", "step2"], log_type="custom_log", interval=3)

        Returns
        -------
        PipelineRun:
            Pipeline run instance (self).
        """
        logging = self.logs(log_type=log_type)

        steps_to_monitor = list(set(steps or ()) | set(args))

        try:
            return self.__stream_log(
                logging,
                steps_to_monitor,
                interval,
                log_type,
            )
        except KeyboardInterrupt:
            print("Stop watching logs.")
            pass

    def __stream_log(
        self,
        logging: ConsolidatedLog,
        pipeline_steps: List = None,
        interval: float = LOG_INTERVAL,
        log_type: str = None,
    ) -> "PipelineRun":
        """Stream logs from OCI pipeline backends.

        Parameters
        ----------
        logging : ConsolidatedLog.
            The ConsolidatedLog instance.
        pipeline_steps: list
            A list of pipeline step name.
        interval : float
            Time interval in seconds between each request to update the logs.
        log_type : str
            The log type.

        Returns
        -------
        PipelineRun:
            Pipeline run instance (self).
        """
        print(f"Pipeline OCID: {self.pipeline_id}")
        print(f"Pipeline Run OCID: {self.id}")

        if self.time_accepted:
            count = logging.stream(
                interval=interval,
                stop_condition=self._stop_condition,
                time_start=self.time_accepted,
                log_filter=self._build_filter_expression(pipeline_steps, log_type),
            )
            if not count:
                print(
                    "No logs in the last 14 days. Please set time_start to see older logs."
                )

        return self.sync()

    def _build_filter_expression(self, steps: List = [], log_type: str = None) -> str:
        """Builds query expression for logs that are generated by pipeline run and job run.
            The query expression consists of two parts:
            1. Logs that are generated by pipeline run:
                - service and custom logs for CUSTOM_SCRIPT step
                - service log for ML_JOB step
                Format: (source = *<pipeline_run_id> AND ( subject = <pipeline_step_name> OR subject = <pipeline_step_name> OR ...))
            2. Logs that are generated by job run:
                - custom log for ML_JOB step
                Format: source = *<job_run_id> OR source = *<job_run_id> OR source = *<job_run_id> OR ...

            TODO:
                This is a temporary solution, and the real fix will be done after the jobs service add pipleine run details in the log data panel.

        Parameters
        ----------
        steps: list
            A list of pipeline step name.
        log_type : str
            The log type.

        Returns
        -------
        str:
            Query string to search the logs of pipeline.
        """
        sources = []
        subjects = []
        skipped_step_list = []
        for step_run in self.step_runs:
            if not steps or (step_run.step_name in steps):
                step_name = step_run.step_name
                if step_run.step_type == StepType.ML_JOB:
                    if not step_run.job_run_id:
                        skipped_step_list.append(step_run.step_name)
                        continue
                    job_run_id = step_run.job_run_id
                    if log_type == LogType.CUSTOM_LOG:
                        sources.append(f"source = '*{job_run_id}'")
                    elif log_type == LogType.SERVICE_LOG:
                        subjects.append(f"subject = '{step_name}'")
                    else:
                        sources.append(f"source = '*{job_run_id}'")
                        subjects.append(f"subject = '{step_name}'")
                else:
                    subjects.append(f"subject = '{step_name}'")

        if skipped_step_list:
            logger.warning(
                f"ML Jobs: {', '.join(skipped_step_list)} log can't be printed since their job run ids are not known at this time. Please stop and rerun the watch() command again to retrieve the job run ids and print the logs."
            )

        filter_list = []

        # add query for logs that are generated by pipeline run
        if subjects:
            pipeline_log_filters = [f"source = '*{self.id}'"]
            pipeline_log_filters.append("(" + " OR ".join(subjects) + ")")
            filter_list = ["(" + " AND ".join(pipeline_log_filters) + ")"]

        # add query for logs that are generated by job run
        if sources:
            filter_list.extend(sources)

        return " OR ".join(filter_list)

    def _stop_condition(self):
        """Stops the sync once the job is in a terminal state."""
        self.sync()
        return self.lifecycle_state in PIPELINE_RUN_TERMINAL_STATE

    @classmethod
    def list(
        cls, pipeline_id: str, compartment_id: Optional[str] = None, **kwargs
    ) -> List["PipelineRun"]:
        """
        List pipeline runs for a given pipeline.

        Parameters
        ----------
        pipeline_id: str.
            The OCID of pipeline.
        compartment_id: (str, optional). Defaults to None.
            The OCID of compartment.
            If `None`, the value will be taken from the environment variables.
        kwargs
            Additional keyword arguments for filtering pipelines.
            - lifecycle_state: str. Allowed values: "CREATING", "ACTIVE", "DELETING", "FAILED", "DELETED"
            - created_by: str
            - limit: int

        Returns
        -------
        List[PipelineRun]
            The list of pipeline runs.
        """
        PipelineRun.list_resource(compartment_id, pipeline_id=pipeline_id, **kwargs)

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

    def to_yaml(self) -> str:
        """Serializes the object into YAML string.

        Returns
        -------
        str
            YAML stored in a string.
        """
        return yaml.safe_dump(self.to_dict())
