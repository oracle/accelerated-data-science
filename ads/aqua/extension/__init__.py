#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from jupyter_server.utils import url_path_join

from ads.aqua.extension.common_handler import __handlers__ as __common_handlers__
from ads.aqua.extension.deployment_handler import (
    __handlers__ as __deployment_handlers__,
)
from ads.aqua.extension.evaluation_handler import __handlers__ as __eval_handlers__
from ads.aqua.extension.finetune_handler import __handlers__ as __finetune_handlers__
from ads.aqua.extension.model_handler import __handlers__ as __model_handlers__
from ads.aqua.extension.ui_handler import __handlers__ as __ui_handlers__
from ads.aqua.extension.ui_websocket_handler import __handlers__ as __ws_handlers__

__handlers__ = (
    __finetune_handlers__
    + __model_handlers__
    + __common_handlers__
    + __deployment_handlers__
    + __ui_handlers__
    + __eval_handlers__
    + __ws_handlers__
)


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
