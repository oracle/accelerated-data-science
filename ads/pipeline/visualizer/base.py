#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ByteString, Dict, List, Optional, Union

import fsspec
from oci.data_science.models.pipeline_step_run import PipelineStepRun


class PipelineVisualizerError(Exception):   # pragma: no cover
    pass


class StepStatus:
    WAITING = "WAITING"
    ACCEPTED = "ACCEPTED"
    IN_PROGRESS = "IN_PROGRESS"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"
    CANCELING = "CANCELING"
    CANCELED = "CANCELED"
    DELETED = "DELETED"
    SKIPPED = "SKIPPED"


class GraphOrientation:
    TOP_BOTTOM = "TB"
    LEFT_RIGHT = "LR"


WAIT_STATUS = {
    StepStatus.WAITING,
    StepStatus.IN_PROGRESS,
    StepStatus.CANCELING,
}

COMPLETE_STATUS = {
    StepStatus.SUCCEEDED,
    StepStatus.FAILED,
    StepStatus.CANCELED,
    StepStatus.DELETED,
    StepStatus.SKIPPED,
}

FAIL_STATUS = {
    StepStatus.FAILED,
    StepStatus.CANCELED,
    StepStatus.DELETED,
    StepStatus.SKIPPED,
}

STATUS_TEXT_MAP = {
    StepStatus.WAITING: "Waiting",
    StepStatus.ACCEPTED: "Accepted",
    StepStatus.IN_PROGRESS: "In Progress",
    StepStatus.FAILED: "Failed",
    StepStatus.SUCCEEDED: "Succeeded",
    StepStatus.CANCELING: "Canceling",
    StepStatus.CANCELED: "Canceled",
    StepStatus.DELETED: "Deleted",
    StepStatus.SKIPPED: "Skipped",
}

STATUS_COLOR_MAP = {
    StepStatus.WAITING: "#747E7E",
    StepStatus.ACCEPTED: "#F26B1D",
    StepStatus.IN_PROGRESS: "#3E975D",
    StepStatus.FAILED: "#9146C2",
    StepStatus.SUCCEEDED: "#2C6CBF",
    StepStatus.CANCELING: "#D92211",
    StepStatus.CANCELED: "#D92211",
    StepStatus.DELETED: "#D92211",
    StepStatus.SKIPPED: "#D92211",
}


class StepKind:
    ML_JOB = "ML_JOB"
    CUSTOM_SCRIPT = "CUSTOM_SCRIPT"
    PIPELINE = "pipeline"


STEP_KIND_MAP = {
    StepKind.ML_JOB: "Datascience Job",
    StepKind.CUSTOM_SCRIPT: "Custom Script",
    StepKind.PIPELINE: "Pipeline",
}


def _to_utc_native(dt: datetime) -> str:
    """Converts offset-aware datetime to offset-native UTC time.

    Parameters
    ----------
    dt : datetime.datetime
        A datetime object to be converted.
        If dt is offset native, it will be returned as is.

    Returns
    -------
    datetime.datetime
        Offset-native datetime
    """
    if not dt:
        return None

    if dt.tzinfo:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _strfdelta(seconds: int, format: str = "%H:%M:%S") -> str:
    """Converts a datetime.timedelta object to a custom-formatted string.

    Parameters
    ----------
    seconds: int
        Seconds to be converted to a formatted string.
    format: (str, optional). Defaults to `%H:%M:%S`.
        The format argument allows custom formatting to be specified.
    """
    if not seconds:
        return ""
    return datetime.utcfromtimestamp(seconds).strftime(format)


def _replace_special_chars(value: str, repl: Optional[str] = "_") -> str:
    """Returns a string obtained by replacing the all special characters in the string,
    including spaces, by the replacement `repl`.

    Parameters
    ----------
    value: str
        The string that needs to be processed.
    repl: (str, optional). Defaults to `_`.
        The replacement for the special characters.

    Returns
    -------
    str
        The new string.
    """
    return re.sub(f"[{re.escape(string.punctuation)} ]", repl, value)


@dataclass
class RendererItemStatus:
    """Class represents the state of the renderer item."""

    name: str
    kind: str = ""
    time_started: datetime = None
    time_finished: datetime = None
    lifecycle_state: str = ""
    lifecycle_details: str = ""
    _key: str = ""

    def __post_init__(self):
        self.time_started = _to_utc_native(self.time_started)
        self.time_finished = _to_utc_native(self.time_finished)
        self._key = f"{_replace_special_chars(self.name)}_{self.kind}".lower()

    @property
    def key(self) -> str:
        """Key of the item.

        Returns
        -------
        str
            The key of the item.
        """
        return self._key

    @property
    def duration(self) -> int:
        """
        Calculates duration in seconds between `time_started` and `time_finished`.

        Returns
        -------
        int
            The duration in seconds between `time_started` and `time_finished`.
        """
        if not self.time_started:
            return 0
        if not self.time_finished:
            return (datetime.utcnow() - self.time_started).seconds
        return (self.time_finished - self.time_started).seconds

    @classmethod
    def from_pipeline_run(cls, pipeline_run: "PipelineRun") -> "RendererItemStatus":
        """Creates class instance from the PipelineRun object.

        Parameters
        ----------
        pipeline_run: PipelineRun
            The PipelineRun object.

        Returns
        -------
        RendererItemStatus
            Instance of RendererItemStatus.

        """
        return cls(
            name=pipeline_run.pipeline.name,
            kind=pipeline_run.pipeline.kind,
            time_started=pipeline_run.time_accepted or pipeline_run.time_started,
            time_finished=pipeline_run.time_finished,
            lifecycle_state=pipeline_run.lifecycle_state,
            lifecycle_details=pipeline_run.lifecycle_details,
        )

    @classmethod
    def from_pipeline_step_run(
        cls, pipeline_step_run: PipelineStepRun
    ) -> "RendererItemStatus":
        """Creates class instance from the PipelineStepRun object.

        Parameters
        ----------
        pipeline_run: PipelineStepRun
            The PipelineStepRun object.

        Returns
        -------
        RendererItemStatus
            Instance of RendererItemStatus.

        """
        return cls(
            name=pipeline_step_run.step_name,
            kind=pipeline_step_run.step_type,
            time_started=pipeline_step_run.time_started,
            time_finished=pipeline_step_run.time_finished,
            lifecycle_state=pipeline_step_run.lifecycle_state,
            lifecycle_details=pipeline_step_run.lifecycle_details,
        )

    @staticmethod
    def format_datetime(value: datetime, format="%Y-%m-%d %H:%M:%S") -> str:
        """Converts datetime object into a given format in string

        Parameters
        ----------
        dt: datetime.datetime
            Datetime object to be formated.

        Returns
        -------
        str:
            A timestamp in a string format.
        """
        if not value:
            return None

        return value.strftime(format)

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass
class RendererItem:
    name: str
    kind: str = ""
    spec: Union["Pipeline", "PipelineStep"] = None
    _key: str = ""

    def __post_init__(self):
        self._key = f"{_replace_special_chars(self.name)}_{self.kind}".lower()

    @property
    def key(self) -> str:
        """Key of the item.

        Returns
        -------
        str
            The key of the item.
        """
        return self._key

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return hash(self) == hash(other)


class PipelineRenderer(ABC):
    """The base class responsible for the vizualizing a pipleine."""

    @abstractmethod
    def render(
        self,
        steps: List[RendererItem],
        deps: Dict[str, List[RendererItem]],
        step_status: Dict[str, List[RendererItemStatus]] = None,
        **kwargs,
    ):
        """Renders pipeline run."""
        pass

    def _write_to_file(
        self, content: Union[str, ByteString], mode: str, uri: str, **kwargs
    ) -> None:
        """Writes string into location specified by uri.

        Parameters
        ----------
        content: Union[str, ByteString]
            The content to be saved.
        mode: str
            The file write mode.
        uri: (string)
            URI location to save content.

        kwargs
        ------
        keyword arguments to be passed into fsspec.open(). For OCI object storage, this should be config="path/to/.oci/config".
            For other storage connections consider e.g. host, port, username, password, etc.

        Returns
        -------
        None
            Nothing.
        """
        with fsspec.open(uri, f"w{mode}", **kwargs) as f:
            f.write(content)

    def save_to(self, *args, **kwargs) -> str:
        """Saves the pipeline visualization to the provided format."""
        raise NotImplementedError("`.save_to()` is not implemented.")


class PipelineVisualizer:
    """PipelineVisualizer class to visualize pipeline in text or graph.

    Attributes
    ----------
    pipeline: Pipeline
        Pipeline instance.
    pipeline_run: PipelineRun
        PipelineRun instance.
    steps: List[RendererItem]
        A list of RendererItem objects.
    deps: Dict[str, List[RendererItem]]
        A dictionary mapping the key of a RendererItem to a list of
        RendererItem that this step depends on.
    step_status: Dict[str, RendererItemStatus], defaults to None.
        A dictionary mapping the key of a RendererItem to its current
        status.
    """

    def __init__(
        self,
        pipeline: "Pipeline" = None,
        pipeline_run: "PipelineRun" = None,
        renderer: PipelineRenderer = None,
    ):
        """Initialize a PipelineVisualizer object.

        Parameters
        ----------
        pipeline: Pipeline
            Pipeline instance.
        pipeline_run: PipelineRun
            PipelineRun instance.
        renderer: PipelineRenderer
            Renderer used to visualize pipeline in text or graph.
        """
        self.pipeline = None
        self.pipeline_run = None
        self.renderer = None
        self.steps = []
        self.deps = {}
        self.step_status = {}

        if pipeline:
            self.with_pipeline(pipeline)
        if pipeline_run:
            self.with_pipeline_run(pipeline_run)
        if renderer:
            self.with_renderer(renderer)

    def with_renderer(self, value: PipelineRenderer) -> "PipelineVisualizer":
        """
        Add renderer to visualize pipeline.

        Parameters
        ----------
        value: object
            Renderer used to visualize pipeline in text or graph.

        Returns
        -------
        PipelineVisualizer
            The PipelineVisualizer instance.

        Raises
        ------
        PipelineVisualizerError
            If `renderer` not specified.
        """
        if not value:
            raise PipelineVisualizerError("The `renderer` must be specified.")

        self.renderer = value
        return self

    def with_pipeline(self, value: "Pipeline") -> "PipelineVisualizer":
        """
        Adds a Pipeline instance to be rendered.

        Parameters
        ----------
        value: Pipeline
            Pipeline instance.

        Returns
        -------
        PipelineVisualizer
            The PipelineVisualizer instance.

        Raises
        ------
        PipelineVisualizerError
            If `pipeline` not specified.
        """
        if not value:
            raise PipelineVisualizerError("The `pipeline` must be specified.")

        self.pipeline = value
        pipeline_render_item = RendererItem(
            name=self.pipeline.name, kind=self.pipeline.kind, spec=self.pipeline
        )
        self.steps = [pipeline_render_item]
        self.deps = {pipeline_render_item.key: []}

        if self.pipeline.step_details:
            render_item_map = {
                step.name: RendererItem(name=step.name, kind=step.kind, spec=step)
                for step in self.pipeline.step_details
            }

            for step in self.pipeline.step_details:
                self.steps.append(render_item_map[step.name])
                if step.depends_on:
                    depends_on = [
                        render_item_map[step_name] for step_name in step.depends_on
                    ]
                else:
                    depends_on = [pipeline_render_item]

                self.deps[render_item_map[step.name].key] = depends_on

        return self

    def with_pipeline_run(self, value: "PipelineRun") -> "PipelineVisualizer":
        """
        Adds a PipelineRun instance to be rendered.

        Parameters
        ----------
        value: PipelineRun
            PipelineRun instance.

        Returns
        -------
        PipelineVisualizer
            The PipelineVisualizer instance.

        Raises
        ------
        PipelineVisualizerError
            If `pipeline run` not specified.
        """
        if not value:
            raise PipelineVisualizerError("The `pipeline run` must be specified.")

        self.pipeline_run = value
        self.step_status = {}

        if self.pipeline_run:
            render_item_status = RendererItemStatus.from_pipeline_run(self.pipeline_run)
            self.step_status[render_item_status.key] = render_item_status
            if self.pipeline_run.step_runs:
                for step_run in self.pipeline_run.step_runs:
                    render_item_status = RendererItemStatus.from_pipeline_step_run(
                        step_run
                    )
                    self.step_status[render_item_status.key] = render_item_status

        return self

    def render(self, rankdir: str = GraphOrientation.TOP_BOTTOM):
        """
        Renders pipeline step status.

        Parameters
        ----------
        rankdir: str, default to "TB".
            Direction of the rendered graph; allowed Values are {"TB", "LR"}.

        Returns
        -------
        None

        Raises
        ------
        PipelineVisualizerError
            If `pipeline` or `renderer` not specified.
        """
        if not (self.steps and self.deps and self.renderer):
            raise PipelineVisualizerError("The `pipeline` must be specified.")

        self.renderer.render(
            steps=self.steps,
            deps=self.deps,
            step_status=self.step_status,
            rankdir=rankdir,
        )

    def to_svg(
        self, uri: str = None, rankdir: str = GraphOrientation.TOP_BOTTOM, **kwargs
    ) -> str:
        """
        Renders pipeline as graph in SVG string.

        Parameters
        ----------
        uri: (string, optional). Defaults to None.
            URI location to save the SVG string.
        rankdir: str, default to "TB".
            Direction of the rendered graph; allowed Values are {"TB", "LR"}.

        Returns
        -------
        str
            Graph in svg format.

        Raises
        ------
        PipelineVisualizerError
            If `pipeline` or `renderer` not specified.
        """
        if not (self.steps and self.deps and self.renderer):
            raise PipelineVisualizerError("The `pipeline` must be specified.")

        return self.renderer.save_to(
            steps=self.steps,
            deps=self.deps,
            step_status=self.step_status,
            rankdir=rankdir,
            uri=uri,
            format="svg",
            **kwargs,
        )
