#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from datetime import datetime
from unittest import SkipTest
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

try:
    from ads.pipeline.ads_pipeline import Pipeline
    from ads.pipeline.ads_pipeline_run import PipelineRun, ShowMode
    from ads.pipeline.visualizer.base import (
        PipelineVisualizer,
        RendererItem,
        RendererItemStatus,
        GraphOrientation,
    )
    from ads.pipeline.visualizer.graph_renderer import PipelineGraphRenderer
except (ImportError, AttributeError) as e:
    raise SkipTest("OCI MLPipeline is not available. Skipping the MLPipeline tests.")


class TestPipelineRunVisualizer:
    """Tests pipeline run visualizer."""

    @classmethod
    def setup_class(cls):
        cls.mock_datetime = datetime.now()
        cls.curdir = os.path.dirname(os.path.abspath(__file__))
        cls.artifact_dir = os.path.join(cls.curdir, "artifact")

        cls.mock_pipeline = Pipeline.from_yaml(
            uri=os.path.join(cls.artifact_dir, "sample_pipeline.yaml")
        )

        with patch.object(Pipeline, "from_ocid") as mock_pipeline_from_ocid:
            with patch.object(
                PipelineRun,
                "pipeline",
                new_callable=PropertyMock,
                return_value=cls.mock_pipeline,
            ):
                mock_pipeline_from_ocid.return_value = cls.mock_pipeline

                cls.mock_pipeline_run = PipelineRun(
                    **{
                        "display_name": cls.mock_pipeline.name,
                        "lifecycle_state": "ACCEPTED",
                        "lifecycle_details": "Test Lifecycle details.",
                        "time_started": cls.mock_datetime,
                        "time_finished": cls.mock_datetime,
                    }
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
        with patch.object(
            PipelineRun,
            "pipeline",
            new_callable=PropertyMock,
            return_value=self.mock_pipeline,
        ):
            self.mock_renderer = PipelineGraphRenderer()

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

    @pytest.mark.parametrize(
        "input_data",
        [
            {
                "mode": ShowMode.GRAPH,
                "wait": True,
                "rankdir": GraphOrientation.LEFT_RIGHT,
            },
            {
                "mode": ShowMode.GRAPH,
                "wait": False,
                "rankdir": GraphOrientation.TOP_BOTTOM,
            },
            {
                "mode": ShowMode.TEXT,
                "wait": True,
                "rankdir": GraphOrientation.LEFT_RIGHT,
            },
            {
                "mode": ShowMode.TEXT,
                "wait": False,
                "rankdir": GraphOrientation.TOP_BOTTOM,
            },
        ],
    )
    @patch.object(PipelineRun, "sync")
    @patch.object(PipelineVisualizer, "render")
    @patch.object(PipelineRun, "_show")
    def test_show_one(
        self, mock__show, mock_render, mock_pipeline_run_sync, input_data
    ):
        """Tests rendering pipeline run in a graph with wait option."""
        with patch.object(
            PipelineRun,
            "pipeline",
            new_callable=PropertyMock,
            return_value=self.mock_pipeline,
        ):
            self.mock_pipeline_run.show(**input_data)
            if input_data["wait"]:
                mock_render.assert_not_called()

                mock__show.assert_called_with(
                    self.mock_pipeline_run._graphViz
                    if input_data["mode"] == ShowMode.GRAPH
                    else self.mock_pipeline_run._textViz,
                    rankdir=input_data["rankdir"],
                )

            else:
                mock_render.assert_called_with(rankdir=input_data["rankdir"])
                mock__show.assert_not_called()

            mock_pipeline_run_sync.assert_called()

    @patch.object(PipelineRun, "sync")
    @patch.object(PipelineVisualizer, "to_svg")
    def test_to_svg(self, mock_to_svg, mock_pipeline_run_sync):
        """Tests rendering pipeline run graph into SVG."""
        with patch.object(
            PipelineRun,
            "pipeline",
            new_callable=PropertyMock,
            return_value=self.mock_pipeline,
        ):
            uri = "test_uri"
            rankdir = "test_rankdir"
            kwargs = {"field": "value"}
            self.mock_pipeline_run.to_svg(uri=uri, rankdir=rankdir, **kwargs)
            mock_to_svg.assert_called_with(uri=uri, rankdir=rankdir, **kwargs)
            mock_pipeline_run_sync.assert_called()
