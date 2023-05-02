#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import unittest
from unittest import mock

import IPython
import pytest

try:
    from ads.pipeline.ads_pipeline_step import PipelineStep
    from ads.pipeline.visualizer.base import RendererItem, RendererItemStatus
    from ads.pipeline.visualizer.graph_renderer import PipelineGraphRenderer
    from ads.pipeline.visualizer.text_renderer import PipelineTextRenderer
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest(
        "OCI MLPipeline is not available. Skipping the MLPipeline tests."
    )


class TestPipelineVisualizer:
    def setup_method(self):
        self.MOCK_STEPS = [
            RendererItem(
                name="TestPipelineStepOne",
                kind="ML_JOB",
                spec=PipelineStep(name="TestPipelineStepOne"),
            ),
            RendererItem(
                name="TestPipelineStepTwo",
                kind="ML_JOB",
                spec=PipelineStep(name="TestPipelineStepTwo"),
            ),
        ]
        self.MOCK_DEPS = {
            self.MOCK_STEPS[0].key: [],
            self.MOCK_STEPS[1].key: [self.MOCK_STEPS[0]],
        }

        self.MOCK_status = {
            self.MOCK_STEPS[0].key: RendererItemStatus(
                name=self.MOCK_STEPS[0].name,
                lifecycle_state="ACCEPTED",
                kind=self.MOCK_STEPS[0].kind,
            ),
            self.MOCK_STEPS[1].key: RendererItemStatus(
                name=self.MOCK_STEPS[1].name,
                lifecycle_state="WAITING",
                kind=self.MOCK_STEPS[1].kind,
            ),
        }

    def test_GraphRenderer_show_status_without_status_fail(self):
        with pytest.raises(ValueError):
            PipelineGraphRenderer(True).render(self.MOCK_STEPS, self.MOCK_DEPS)

    def test_TextRenderer_success(self):
        result = PipelineTextRenderer().render(
            self.MOCK_STEPS, self.MOCK_DEPS, self.MOCK_status
        )
        assert result == None

    @mock.patch.object(IPython.core.display, "display")
    def test_GraphRenderer_no_status_success(self, mock_display):
        PipelineGraphRenderer().render(self.MOCK_STEPS, self.MOCK_DEPS)
        mock_display.assert_called_once()

    @mock.patch.object(IPython.core.display, "display")
    def test_GraphRenderer_status_success(self, mock_display):
        PipelineGraphRenderer(True).render(
            self.MOCK_STEPS, self.MOCK_DEPS, self.MOCK_status
        )
        mock_display.assert_called_once()

    def test_GraphRenderer_to_svg(self):
        test_result = PipelineGraphRenderer().save_to(self.MOCK_STEPS, self.MOCK_DEPS)
        assert "<svg" in test_result.decode("utf-8")

    def test_GraphRenderer_to_svg_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = os.path.join(temp_dir, "test.svg")
            test_result = PipelineGraphRenderer().save_to(
                self.MOCK_STEPS, self.MOCK_DEPS, uri=result_file
            )
            with open(result_file, "rb") as svg_file:
                file_result = svg_file.read()
            assert test_result == file_result
