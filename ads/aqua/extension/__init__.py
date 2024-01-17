#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from .model_handler import AquaModelHandler
from .job_handler import AquaFineTuneHandler


def _jupyter_server_extension_paths():
    return [{"module": "ads.aqua.extension"}]


def load_jupyter_server_extension(serverapp):
    """
    This function is called when the extension is loaded.
    """
    handlers = [
        ("/aqua/model/?([^/]*)", AquaModelHandler),
        ("/aqua/finetuning/?([^/]*)", AquaFineTuneHandler),
    ]
    serverapp.web_app.add_handlers(".*$", handlers)
