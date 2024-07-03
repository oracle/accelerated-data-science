#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.evaluation.entities import CreateAquaEvaluationDetails
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.extension.utils import validate_function_parameters
from ads.config import COMPARTMENT_OCID


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
        """Handles post request for the evaluation APIs

        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid.
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        validate_function_parameters(
            data_class=CreateAquaEvaluationDetails, input_data=input_data
        )

        self.finish(
            # TODO: decide what other kwargs will be needed for create aqua evaluation.
            AquaEvaluationApp().create(
                create_aqua_evaluation_details=(
                    CreateAquaEvaluationDetails(**input_data)
                )
            )
        )

    @handle_exceptions
    def put(self, eval_id):
        """Handles PUT request for the evaluation APIs"""
        eval_id = eval_id.split("/")[0]
        return self.finish(AquaEvaluationApp().cancel(eval_id))

    @handle_exceptions
    def delete(self, eval_id):
        return self.finish(AquaEvaluationApp().delete(eval_id))

    def read(self, eval_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaEvaluationApp().get(eval_id))

    def list(self):
        """List Aqua models."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        return self.finish(AquaEvaluationApp().list(compartment_id))

    def get_default_metrics(self):
        """Lists supported metrics for evaluation."""
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


class AquaEvaluationConfigHandler(AquaAPIhandler):
    """Handler for Aqua Evaluation Config REST APIs."""

    @handle_exceptions
    def get(self, model_id):
        """Handle GET request."""

        return self.finish(AquaEvaluationApp().load_evaluation_config(model_id))


__handlers__ = [
    ("evaluation/config/?([^/]*)", AquaEvaluationConfigHandler),
    ("evaluation/?([^/]*)", AquaEvaluationHandler),
    ("evaluation/?([^/]*/report)", AquaEvaluationReportHandler),
    ("evaluation/?([^/]*/metrics)", AquaEvaluationMetricsHandler),
    ("evaluation/?([^/]*/status)", AquaEvaluationStatusHandler),
    ("evaluation/?([^/]*/cancel)", AquaEvaluationHandler),
]
