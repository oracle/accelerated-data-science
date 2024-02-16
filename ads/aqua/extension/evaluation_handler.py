#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.aqua.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.base_handler import AquaAPIhandler


class AquaEvaluationHandler(AquaAPIhandler):
    """Handler for Aqua Model Evaluation REST APIs."""

    @handle_exceptions
    def get(self, eval_id=""):
        """Handle GET request."""
        if not eval_id:
            return self.list()
        return self.read(eval_id)

    @handle_exceptions
    def post(self, *args, **kwargs):
        """
        Handles post request for the deployment APIs
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        # TODO:  create
        pass

    @handle_exceptions
    def read(self, eval_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaEvaluationApp().get(eval_id))

    @handle_exceptions
    def list(self):
        """List Aqua models."""
        compartment_id = self.get_argument("compartment_id", default=None)
        # project_id is no needed.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(AquaEvaluationApp().list(compartment_id, project_id))


class AquaEvaluationReportHandler(AquaAPIhandler):
    """Handler for Aqua Evaluation report REST APIs."""

    @handle_exceptions
    def get(self, eval_id):
        """Handle GET request."""
        eval_id = eval_id.split("/")[0]
        return self.finish(AquaEvaluationApp().download_report(eval_id))


class AquaEvaluationMetaHandler(AquaAPIhandler):
    """Handler for Aqua Evaluation metadata REST APIs."""

    @handle_exceptions
    def get(self, eval_id):
        """Handle GET request."""
        eval_id = eval_id.split("/")[0]
        return self.finish(AquaEvaluationApp().load_params(eval_id))


__handlers__ = [
    ("evaluation/?([^/]*)", AquaEvaluationHandler),
    ("evaluation/?([^/]*/report)", AquaEvaluationReportHandler),
    ("evaluation/?([^/]*/meta)", AquaEvaluationMetaHandler),
]
