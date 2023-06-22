#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import os
import unittest
from ads.common.oci_datascience import DSCNotebookSession
from tests.integration.config import secrets

logger = logging.getLogger(__name__)


class DSCNotebookSessionTestCase(unittest.TestCase):
    COMPARTMENT_ID = secrets.common.COMPARTMENT_ID

    @classmethod
    def setUpClass(cls) -> None:
        # Save existing env vars
        cls.existing_env = os.environ
        # Set compartment ID
        os.environ["NB_SESSION_COMPARTMENT_OCID"] = cls.COMPARTMENT_ID
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        # Restore env vars
        os.environ = cls.existing_env
        return super().tearDownClass()

    def test_list_notebook_and_get_subnet_id(self):
        # List the notebooks in the default compartment and check the subnet ID of the first notebook in the list.
        notebooks = DSCNotebookSession.list_resource()
        self.assertIsInstance(notebooks, list)
        notebooks = [
            notebook
            for notebook in notebooks
            if notebook.notebook_session_configuration_details
        ]
        if len(notebooks) < 1:
            logger.warning("No notebook found in the compartment.")
            return
        subnet_id = notebooks[0].notebook_session_configuration_details.subnet_id
        self.assertTrue(str(subnet_id).startswith("ocid1.subnet.oc1.iad"))
        # Get the notebook object from OCID
        notebook = DSCNotebookSession.from_ocid(notebooks[0].id)
        # The two notebook objects should have the same subnet ID.
        self.assertTrue(
            subnet_id, notebook.notebook_session_configuration_details.subnet_id
        )
