#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from datetime import datetime
from unittest import SkipTest
from unittest.mock import MagicMock, patch

import pytest

try:
    from ads.pipeline import Pipeline
    from ads.pipeline.visualizer.base import (
        PipelineVisualizer,
        PipelineVisualizerError,
        RendererItem,
        RendererItemStatus,
        _replace_special_chars,
    )
    from ads.pipeline.visualizer.text_renderer import PipelineTextRenderer
except (ImportError, AttributeError) as e:
    raise SkipTest("OCI MLPipeline is not available. Skipping the MLPipeline tests.")


class TestPipelineVisualizer:
    """Tests for PipelineVisualizer class to visualize pipeline in text or graph."""

    @classmethod
    def setup_class(cls):
        cls.mock_datetime = datetime.now()
        cls.curdir = os.path.dirname(os.path.abspath(__file__))
        cls.artifact_dir = os.path.join(cls.curdir, "artifact")

        cls.mock_pipeline = Pipeline.from_yaml(
            uri=os.path.join(cls.artifact_dir, "sample_pipeline.yaml")
        )

        cls.mock_pipeline_run = MagicMock(
            name=cls.mock_pipeline.name,
            lifecycle_state="ACCEPTED",
            lifecycle_details="Test Lifecycle details.",
            time_started=cls.mock_datetime,
            time_finished=cls.mock_datetime,
            pipeline=cls.mock_pipeline,
        )
        cls.mock_pipeline_run.step_runs = [
            MagicMock(
                step_name="PipelineStepOne",
                step_type="ML_JOB",
                lifecycle_state="ACCEPTED",
                lifecycle_details="Test Lifecycle details.",
                time_started=cls.mock_datetime,
                time_finished=cls.mock_datetime,
            ),
            MagicMock(
                step_name="PipelineStepTwo",
                step_type="ML_JOB",
                lifecycle_state="WAITING",
                lifecycle_details="Test Lifecycle details.",
                time_started=cls.mock_datetime,
                time_finished=cls.mock_datetime,
            ),
        ]

    def setup_method(self):

        self.mock_renderer = PipelineTextRenderer()

        # mocking steps and steps dependencies
        pipeline_render_item = RendererItem(
            name=self.mock_pipeline.name,
            kind=self.mock_pipeline.kind,
            spec=self.mock_pipeline,
        )
        self.mock_steps = [pipeline_render_item]
        self.mock_deps = {pipeline_render_item.key: []}

        if self.mock_pipeline.step_details:
            render_item_map = {
                step.name: RendererItem(name=step.name, kind=step.kind, spec=step)
                for step in self.mock_pipeline.step_details
            }

            for step in self.mock_pipeline.step_details:
                self.mock_steps.append(render_item_map[step.name])
                if step.depends_on:
                    depends_on = [
                        render_item_map[step_name] for step_name in step.depends_on
                    ]
                else:
                    depends_on = [pipeline_render_item]

                self.mock_deps[render_item_map[step.name].key] = depends_on

        # mocking statuses
        self.mock_step_status = {}
        render_item_status = RendererItemStatus.from_pipeline_run(
            self.mock_pipeline_run
        )
        self.mock_step_status[render_item_status.key] = render_item_status
        for step_run in self.mock_pipeline_run.step_runs:
            render_item_status = RendererItemStatus.from_pipeline_step_run(step_run)
            self.mock_step_status[render_item_status.key] = render_item_status

        # mocking visualizer
        self.mock_visualizer = PipelineVisualizer(
            pipeline=self.mock_pipeline,
            pipeline_run=self.mock_pipeline_run,
            renderer=self.mock_renderer,
        )

    def test_init(self):
        """Ensures pipeline visualizer can be initialized."""
        assert self.mock_visualizer.pipeline == self.mock_pipeline
        assert self.mock_visualizer.pipeline_run == self.mock_pipeline_run
        assert self.mock_visualizer.renderer == self.mock_renderer
        assert self.mock_visualizer.deps == self.mock_deps
        assert self.mock_visualizer.steps == self.mock_steps
        assert self.mock_visualizer.step_status == self.mock_step_status

    def test_with_renderer(self):
        """Tests adding renderer to visualize pipeline."""
        test_renderer = PipelineTextRenderer()
        result = self.mock_visualizer.with_renderer(test_renderer)
        assert self.mock_visualizer.renderer == test_renderer
        assert result == self.mock_visualizer

    def test_with_pipeline(self):
        """Tests adding pipeline."""
        result = self.mock_visualizer.with_pipeline(self.mock_pipeline)
        assert self.mock_visualizer.pipeline == self.mock_pipeline
        assert self.mock_visualizer.deps == self.mock_deps
        assert self.mock_visualizer.steps == self.mock_steps
        assert result == self.mock_visualizer

    def test_with_pipeline_run(self):
        """Tests adding pipeline run."""
        result = self.mock_visualizer.with_pipeline_run(self.mock_pipeline_run)
        assert self.mock_visualizer.pipeline_run == self.mock_pipeline_run
        assert self.mock_visualizer.step_status == self.mock_step_status
        assert result == self.mock_visualizer

    def test_to_svg(self):
        """Tests rendering pipeline as graph in SVG string."""
        with pytest.raises(PipelineVisualizerError):
            self.mock_visualizer.with_renderer(None)
            self.mock_visualizer.to_svg()

        with patch.object(PipelineTextRenderer, "save_to") as mock_save_to:
            self.mock_visualizer.with_renderer(self.mock_renderer)
            self.mock_visualizer.to_svg(
                uri="test_uri", rankdir="LR", extra_prop="prop_val"
            )

            mock_save_to.assert_called_with(
                steps=self.mock_steps,
                deps=self.mock_deps,
                step_status=self.mock_step_status,
                rankdir="LR",
                uri="test_uri",
                format="svg",
                **{"extra_prop": "prop_val"},
            )


class TestVisualizerBase:
    """Tests common methods of the pipeline visualizer base module."""

    @pytest.mark.parametrize(
        "test_value, repl, expected_result",
        [
            ("asd*()@wer", "_", "asd____wer"),
            ("asd*()@wer 12 r", "#", "asd####wer#12#r"),
        ],
    )
    def test__replace_special_chars(self, test_value, repl, expected_result):
        assert _replace_special_chars(test_value, repl) == expected_result
