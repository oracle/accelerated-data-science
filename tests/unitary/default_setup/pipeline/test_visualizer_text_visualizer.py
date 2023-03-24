#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest import SkipTest
from unittest.mock import patch

try:
    from ads.pipeline.ads_pipeline_step import PipelineStep
    from ads.pipeline.visualizer.base import RendererItem, RendererItemStatus
    from ads.pipeline.visualizer.text_renderer import PipelineTextRenderer
except (ImportError, AttributeError) as e:
    raise SkipTest("OCI MLPipeline is not available. Skipping the MLPipeline tests.")


class TestPipelineTextRenderer:
    """Tests Pipeline Steps Text Visualizer."""

    def setup_method(self):
        self.mock_renderer = PipelineTextRenderer()

    def test__render(self):
        """Tests rendering pipeline steps in text."""

        result = self.mock_renderer._render(steps=None, deps=None, step_status=None)
        assert result == []
        result = self.mock_renderer._render(
            steps=None, deps=None, step_status={"key": "value"}
        )
        assert result == []

        steps = [
            RendererItem(
                name="step1", kind="dataScienceJob", spec=PipelineStep(name="step1")
            ),
        ]

        result = self.mock_renderer._render(
            steps=steps,
            deps={},
            step_status={
                steps[0].key: RendererItemStatus(
                    name="step1", lifecycle_state="ACCEPTED", kind="dataScienceJob"
                )
            },
        )
        assert result == [{"Status": "Accepted", "Step": "step1: "}]

        steps = [
            RendererItem(
                name="step1", kind="dataScienceJob", spec=PipelineStep(name="step1")
            ),
            RendererItem(
                name="step2", kind="dataScienceJob", spec=PipelineStep(name="step2")
            ),
        ]
        result = self.mock_renderer._render(
            steps=steps,
            deps={steps[0].key: [], steps[1].key: [steps[0]]},
            step_status={
                steps[0].key: RendererItemStatus(
                    name="step1", lifecycle_state="ACCEPTED", kind="dataScienceJob"
                ),
                steps[1].key: RendererItemStatus(
                    name="step2", lifecycle_state="WAITING", kind="dataScienceJob"
                ),
            },
        )

        assert result == [
            {"Status": "Accepted", "Step": "step1: "},
            {"Status": "Waiting [step1]", "Step": "step2: "},
        ]

    @patch.object(PipelineTextRenderer, "_render")
    def test_render(self, mock_render):
        """Tests rendering pipeline steps in text."""
        test_renderer = PipelineTextRenderer()

        mock_steps = [
            RendererItem(
                name="step1", kind="dataScienceJob", spec=PipelineStep(name="step1")
            ),
            RendererItem(
                name="step2", kind="dataScienceJob", spec=PipelineStep(name="step2")
            ),
        ]

        mock_deps = {mock_steps[0].key: [], "step2": [mock_steps[0]]}
        mock_step_status = {
            mock_steps[0].key: RendererItemStatus(
                name="step1", lifecycle_state="ACCEPTED", kind="dataScienceJob"
            ),
            mock_steps[1].key: RendererItemStatus(
                name="step2", lifecycle_state="WAITING", kind="dataScienceJob"
            ),
        }

        test_renderer.render(
            steps=mock_steps,
            deps=mock_deps,
            step_status=mock_step_status,
        )

        mock_render.assert_called_with(
            steps=mock_steps, deps=mock_deps, step_status=mock_step_status
        )
