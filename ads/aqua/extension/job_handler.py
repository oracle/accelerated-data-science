#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from tornado.web import HTTPError
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.job import AquaFineTuningApp


class AquaFineTuneHandler(AquaAPIhandler):
    """Handler for Aqua fine-tuning job REST APIs."""

    def get(self, job_id=""):
        """Handle GET request."""
        if not job_id:
            return self.list()
        return self.read(job_id)

    def read(self, job_id):
        """Read the information of an Aqua Job."""
        if job_id == "x1":
            raise HTTPError(400, "Invalid JOB ID.")
        if job_id == "x2":
            raise HTTPError(500, "API call error.")
        return self.finish(AquaFineTuningApp().get(job_id))

    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        compartment_id = self.get_argument("compartment_id")
        # project_id is optional.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(AquaFineTuningApp().list(compartment_id, project_id))
