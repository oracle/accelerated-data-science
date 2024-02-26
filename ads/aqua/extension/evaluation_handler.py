#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse

from ads.aqua.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.base_handler import AquaAPIhandler


class AquaEvaluationHandler(AquaAPIhandler):
    """Handler for Aqua Model Evaluation REST APIs."""

    @handle_exceptions
    def get(self, eval_id=""):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/evaluation/metrics"):
            return self.get_default_metrics()
        if not eval_id:
            return self.list()
        return self.read(eval_id)

    @handle_exceptions
    def post(self, *args, **kwargs):
        """Handles post request for the evaluation APIs"""
        self.finish(AquaEvaluationApp().create())

    @handle_exceptions
    def put(self, eval_id):
        """Handles PUT request for the evaluation APIs"""
        self.finish(
            {
                "evaluation_id": eval_id,
                "status": "CANCELLED",
                "time_accepted": "2024-02-15 20:18:34.225000+00:00",
            }
        )

    @handle_exceptions
    def delete(self, eval_id):
        self.finish(
            {
                "evaluation_id": eval_id,
                "status": "DELETING",
                "time_accepted": "2024-02-15 20:18:34.225000+00:00",
            }
        )

    def read(self, eval_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaEvaluationApp().get(eval_id))

    def list(self):
        """List Aqua models."""
        compartment_id = self.get_argument("compartment_id", default=None)
        # project_id is no needed.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(AquaEvaluationApp().list(compartment_id, project_id))

    def get_default_metrics(self, **kwargs):
        """Lists supported metrics."""
        return self.finish(AquaEvaluationApp().get_supported_metrics())


class AquaEvaluationStatusHandler(AquaAPIhandler):
    """Handler for Aqua Evaluation status REST APIs."""

    @handle_exceptions
    def get(self, eval_id):
        """Handle GET request."""
        eval_id = eval_id.split("/")[0]
        return self.finish(AquaEvaluationApp().get_status(eval_id))


class AquaEvaluationReportHandler(AquaAPIhandler):
    """Handler for Aqua Evaluation report REST APIs."""

    @handle_exceptions
    def get(self, eval_id):
        """Handle GET request."""
        eval_id = eval_id.split("/")[0]
        return self.finish(AquaEvaluationApp().download_report(eval_id))


class AquaEvaluationMetricsHandler(AquaAPIhandler):
    """Handler for Aqua Evaluation metrics REST APIs."""

    @handle_exceptions
    def get(self, eval_id):
        """Handle GET request."""
        eval_id = eval_id.split("/")[0]
        return self.finish(AquaEvaluationApp().load_metrics(eval_id))


__handlers__ = [
    ("evaluation/?([^/]*)", AquaEvaluationHandler),
    ("evaluation/?([^/]*/report)", AquaEvaluationReportHandler),
    ("evaluation/?([^/]*/metrics)", AquaEvaluationMetricsHandler),
    ("evaluation/?([^/]*/status)", AquaEvaluationStatusHandler),
    ("evaluation/?([^/]*/cancel)", AquaEvaluationHandler),
]
