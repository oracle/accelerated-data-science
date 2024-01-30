#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from jupyter_server.utils import url_path_join

from ads.aqua.extension.job_handler import __handlers__ as __job_handlers__
from ads.aqua.extension.model_handler import __handlers__ as __model_handlers__

from ads.aqua.extension.playground_handler import (
    __handlers__ as __playground_handlers__,
)

__handlers__ = __playground_handlers__ + __job_handlers__ + __model_handlers__


def load_jupyter_server_extension(nb_server_app):
    web_app = nb_server_app.web_app
    host_pattern = ".*$"
    route_pattern = url_path_join(web_app.settings["base_url"], "aqua")

    web_app.add_handlers(
        host_pattern,
        [(url_path_join(route_pattern, url), handler) for url, handler in __handlers__],
    )


def _jupyter_server_extension_paths():
    return [{"module": "ads.aqua.extension"}]