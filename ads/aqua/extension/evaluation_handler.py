#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from requests import HTTPError
from ads.aqua.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.aqua.extension.utils import validate_function_parameters
from ads.config import COMPARTMENT_OCID, PROJECT_OCID


class AquaEvaluationHandler(AquaAPIhandler):
    """Handler for Aqua Model Evaluation REST APIs."""

    @handle_exceptions
    def get(self, eval_id=""):
        """Handle GET request."""
        if not eval_id:
            return self.list()
        print(self.xsrf_token)
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
        except Exception:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT)

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)
        
        validate_function_parameters(
            function=AquaEvaluationApp.create, 
            input_data=input_data
        )
        
        try:
            self.finish(
                AquaEvaluationApp().create(
                    evaluation_source_id=input_data.get("evaluation_source_id"),
                    evaluation_name=input_data.get("evaluation_name"),
                    evaluation_description=input_data.get("evaluation_description"),
                    project_id=input_data.get("project_id", PROJECT_OCID),
                    dataset_path=input_data.get("dataset_path"),
                    report_path=input_data.get("report_path"),
                    model_parameters=input_data.get("model_parameters"),
                    shape_name=input_data.get("shape_name"),
                    memory_in_gbs=input_data.get("memory_in_gbs"),
                    ocpus=input_data.get("ocpus"),
                    block_storage_size=input_data.get("block_storage_size"),
                    compartment_id=input_data.get("compartment_id", COMPARTMENT_OCID),
                    experiment_id=input_data.get("experiment_id"), 
                    experiment_name=input_data.get("experiment_name"), 
                    experiment_description=input_data.get("experiment_description"),
                    log_group_id=input_data.get("log_group_id"),
                    log_id=input_data.get("log_id"),
                    metrics=input_data.get("metrics"),
                )
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

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
