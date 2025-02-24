#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from logging import getLogger

from dotenv import load_dotenv

from ads.aqua.server.app import start_server

logger = getLogger(__name__)
config_location = os.path.join(os.getcwd(), ".env")
if os.path.exists(config_location):
    logger.info(f"Loading environment variables from {config_location}")
    load_dotenv(dotenv_path=config_location)
    logger.info("Environment variables loaded successfully")
else:
    logger.warning(
        f"{config_location} not found. Consider using `.env` file to setup default environment variables"
    )

start_server()
