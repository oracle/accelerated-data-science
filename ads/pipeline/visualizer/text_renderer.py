#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List

from ads.pipeline.visualizer.base import (
    COMPLETE_STATUS,
    STATUS_TEXT_MAP,
    PipelineRenderer,
    RendererItem,
    RendererItemStatus,
    StepStatus,
    StepKind,
)
from tabulate import tabulate


class PipelineTextRenderer(PipelineRenderer):
    def render(
        self,
        steps: List[RendererItem],
        deps: Dict[str, List[RendererItem]],
        step_status: Dict[str, RendererItemStatus] = None,
        **kwargs,
    ):
        """
        Render pipeline step status in text.

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

        Returns
        -------
        None
        """
        print(
            tabulate(
                self._render(steps=steps, deps=deps, step_status=step_status, **kwargs),
                headers="keys",
            )
        )

    def _render(
        self,
        steps: List[RendererItem],
        deps: Dict[str, List[str]],
        step_status: Dict[str, RendererItemStatus],
        **kwargs,
    ) -> List[Dict[str, str]]:
        result = []
        steps = steps or []
        deps = deps or {}
        step_status = step_status or {}

        for step in steps:
            status = ""
            if step.kind != StepKind.PIPELINE:
                if step_status:
                    status = STATUS_TEXT_MAP[step_status[step.key].lifecycle_state]
                    if status.lower() == StepStatus.WAITING.lower():
                        unfinished_depended_steps = [
                            substep.name
                            for substep in deps.get(step.key, [])
                            if step_status[substep].lifecycle_state
                            not in COMPLETE_STATUS
                        ]
                        if unfinished_depended_steps == 0:
                            continue
                        status = f"{status} [{', '.join(unfinished_depended_steps)}]"

                result.append({"Step": f"{step.name}: ", "Status": status})

        return result
