#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import itertools
from typing import Dict, List

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.common.extended_enum import ExtendedEnumMeta
from ads.pipeline.visualizer.base import (
    COMPLETE_STATUS,
    FAIL_STATUS,
    STATUS_COLOR_MAP,
    STATUS_TEXT_MAP,
    STEP_KIND_MAP,
    GraphOrientation,
    PipelineRenderer,
    RendererItem,
    RendererItemStatus,
    StepStatus,
    _strfdelta,
)

GRAPH_BOX_COLOR = "#DEDEDE"


class RenderTo(str, metaclass=ExtendedEnumMeta):
    SVG = "svg"
    JPEG = "jpeg"
    PNG = "png"


STEP_WITH_STATUS_LABEL_TEMPLATE = """<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><FONT COLOR="white" POINT-SIZE="14.0" FACE="Helvetica,Arial,sans-serif">{step_name}</FONT></TD></TR>
        <TR BORDER="1"><TD><FONT COLOR="white" POINT-SIZE="11.0" FACE="Helvetica,Arial,sans-serif">{step_kind}</FONT></TD></TR>
        <TR><TD><FONT COLOR="white" POINT-SIZE="11.0"  FACE="Courier New">{status_name}</FONT></TD></TR>
        <TR><TD><FONT COLOR="white" POINT-SIZE="11.0" FACE="Courier New">{duration}&nbsp;</FONT></TD></TR>
        </TABLE>>"""

STEP_LABEL_TEMPLATE = """<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
    <TR><TD><FONT COLOR="white" POINT-SIZE="14.0" FACE="Helvetica,Arial,sans-serif">{step_name}</FONT></TD></TR>
    <TR><TD><FONT COLOR="white" POINT-SIZE="11.0" FACE="Helvetica,Arial,sans-serif">{step_kind}</FONT></TD></TR>
    </TABLE>>"""


class PipelineGraphRenderer(PipelineRenderer):
    def __init__(self, show_status: bool = False):
        """Initialize a PipelineGraphRenderer class.

        Parameters
        ----------
        show_status : bool, defaults to False.
            Whether to display status for steps.

        Returns
        -------
        None
            Nothing.
        """
        super().__init__()
        self.show_status = show_status

    def _add_final_step(self):
        """
        Add final step when rendering pipeline step with status in graph.

        Returns
        -------
        None
        """
        final_step_name = "Done"

        all_steps_set = set([step.key for step in self.steps])
        steps_deps_set = set(itertools.chain(*self.deps.values()))
        final_nodes = list(all_steps_set - steps_deps_set)
        final_step_status = StepStatus.WAITING
        if all(
            single_step_status.lifecycle_state == StepStatus.SUCCEEDED
            for single_step_status in self.step_status.values()
        ):
            final_step_status = StepStatus.SUCCEEDED
        elif any(
            single_step_status.lifecycle_state in FAIL_STATUS
            for single_step_status in self.step_status.values()
        ) and all(
            single_step_status.lifecycle_state in COMPLETE_STATUS
            for single_step_status in self.step_status.values()
        ):
            final_step_status = StepStatus.FAILED

        self.graph.node(
            name=final_step_name,
            label=f'<<FONT POINT-SIZE="16.0" color="white" FACE="Helvetica,Arial,sans-serif"><B>{final_step_name}</B></FONT>>',
            shape="tripleoctagon",
            style="filled, rounded",
            fontsize="18.0",
            color=GRAPH_BOX_COLOR,
            fillcolor=STATUS_COLOR_MAP[final_step_status],
        )

        for node in final_nodes:
            self.graph.edge(node, final_step_name)

    @runtime_dependency(module="graphviz", install_from=OptionalDependency.VIZ)
    def _generate_graph(
        self,
        steps: List[RendererItem],
        deps: Dict[str, List[RendererItem]] = None,
        step_status: Dict[str, RendererItemStatus] = None,
        rankdir: str = GraphOrientation.TOP_BOTTOM,
        **kwargs,
    ):
        """
        Generates Pipeline graph.

        Parameters
        ----------
        steps: List[RendererItem]
            A list of RendererItem objects.
        deps: Dict[str, List[RendererItem]]
            A dictionary mapping the key of a RendererItem to a list of
            RendererItem that this step depends on.
        step_status: Dict[str, RendererItemStatus], defaults to None.
            A dictionary mapping the key of a RendererItem to its current
            status.
        rankdir: str, default to "TB".
            Direction of the rendered graph; allowed Values are {"TB", "LR"}.

        Returns
        -------
        None
        """
        from graphviz import Digraph

        self.steps = steps
        self.deps = deps
        self.step_status = step_status

        if self.show_status and not step_status:
            raise ValueError(
                "`step_status` must be provided to render step status in graph."
            )

        self.graph = Digraph(graph_attr={"rankdir": rankdir})
        self.graph.attr("node", shape="box")

        for step in steps:
            if self.show_status and step_status:
                label = STEP_WITH_STATUS_LABEL_TEMPLATE.format(
                    step_name=step.name,
                    status_name=STATUS_TEXT_MAP[step_status[step.key].lifecycle_state],
                    step_kind=STEP_KIND_MAP[step.kind],
                    duration=_strfdelta(step_status[step.key].duration),
                )
                step_fillcolor = STATUS_COLOR_MAP[step_status[step.key].lifecycle_state]

            else:
                label = STEP_LABEL_TEMPLATE.format(
                    step_name=step.name,
                    step_kind=STEP_KIND_MAP[step.kind],
                )
                step_fillcolor = STATUS_COLOR_MAP[StepStatus.WAITING]

            self.graph.node(
                name=step.key,
                label=label,
                shape=None,
                style="filled, rounded",
                fontsize="11",
                color=GRAPH_BOX_COLOR,
                fillcolor=step_fillcolor,
            )
            for dep in deps[step.key]:
                self.graph.edge(dep.key, step.key)

        if self.show_status and step_status:
            self._add_final_step()

    def render(
        self,
        steps: List[RendererItem],
        deps: Dict[str, List[RendererItem]] = None,
        step_status: Dict[str, RendererItemStatus] = None,
        rankdir: str = GraphOrientation.TOP_BOTTOM,
        **kwargs,
    ):
        """
        Renders Pipeline graph.

        Parameters
        ----------
        steps: List[RendererItem]
            A list of RendererItem objects.
        deps: Dict[str, List[RendererItem]]
            A dictionary mapping the key of a RendererItem to a list of
            RendererItem that this step depends on.
        step_status: Dict[str, RendererItemStatus], defaults to None.
            A dictionary mapping the key of a RendererItem to its current
            status.
        rankdir: str, default to "TB".
            Direction of the rendered graph; allowed Values are {"TB", "LR"}.

        Returns
        -------
        None
        """
        self._generate_graph(
            steps=steps,
            deps=deps,
            step_status=step_status,
            rankdir=rankdir,
        )
        try:
            from IPython.core.display import display

            display(self.graph)
        except:
            pass

    def save_to(
        self,
        steps: List[RendererItem],
        deps: Dict[str, List[RendererItem]] = None,
        step_status: Dict[str, RendererItemStatus] = None,
        rankdir: str = GraphOrientation.TOP_BOTTOM,
        uri: str = None,
        format: str = RenderTo.SVG,
        **kwargs,
    ) -> str:
        """
        Renders pipeline as graph in selected format.

        steps: List[RendererItem]
            A list of RendererItem objects.
        deps: Dict[str, List[RendererItem]]
            A dictionary mapping the key of a RendererItem to a list of
            RendererItem that this step depends on.
        step_status: Dict[str, RendererItemStatus], defaults to None.
            A dictionary mapping the key of a RendererItem to its current
            status.
        rankdir: str, default to "TB".
            Direction of the rendered graph; allowed Values are {"TB", "LR"}.
        uri: (string, optional). Defaults to None.
            URI location to save the SVG string.
        format: (str, optional). Defaults to "svg".
            The format to save the graph.
            Supported formats: "svg", "html".

        Returns
        -------
        str
            Graph in selected format.
        """
        if format not in RenderTo:
            raise ValueError(
                f"Unsupported format: `{format}`. Supported formats: {RenderTo.values()}"
            )

        self._generate_graph(
            steps=steps,
            deps=deps,
            step_status=step_status,
            rankdir=rankdir,
        )
        result = self.graph.pipe(format=format)
        if uri:
            self._write_to_file(content=result, uri=uri, mode="b", **kwargs)
        return result
