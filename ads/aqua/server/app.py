#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from logging import DEBUG, getLogger

import tornado.ioloop
import tornado.web

from ads.aqua.extension import __handlers__

logger = getLogger(__name__)
AQUA_PORT = "AQUA_PORT"
AQUA_HOST = "AQUA_HOST"
AQUA_PROCESS_COUNT = "AQUA_PROCESS_COUNT"
AQUA_CORS_ENABLE = "AQUA_CORS_ENABLE"

URL_PATTERN = r"/aqua/"


def prepare(self):
    self.set_header("Access-Control-Allow-Origin", "*")


def make_app():
    # Patch the prepare method to allow CORS request
    if os.environ.get(AQUA_CORS_ENABLE, "0") == "1":
        for _, handler in __handlers__:
            handler.prepare = prepare
    handlers = [(URL_PATTERN + url, handler) for url, handler in __handlers__]
    # logger.debug(handlers)
    return tornado.web.Application(handlers)


def start_server():
    access_log = getLogger("tornado.access")
    # Set the logging level to DEBUG
    access_log.setLevel(DEBUG)
    app = make_app()
    logger.info("Endpoints:")
    for rule in app.wildcard_router.rules:
        # Depending on the rule type, the route may be stored in different properties.
        # If the rule has a regex matcher, you can get its pattern.
        regex = (
            rule.matcher.regex.pattern
            if hasattr(rule.matcher, "regex")
            else str(rule.matcher)
        )
        print(f"\t\t{regex}")
    server = tornado.httpserver.HTTPServer(app)
    port = int(os.environ.get(AQUA_PORT, 8080))
    host = os.environ.get(AQUA_HOST, "0.0.0.0")
    processes = int(os.environ.get(AQUA_PROCESS_COUNT, 0))
    server.bind(port=port, address=host)
    server.start(processes)
    logger.info(f"Starting the server from directory: {os.getcwd()}")
    logger.info(f"Aqua API server running on http://{host}:{port}")
    tornado.ioloop.IOLoop.current().start()
