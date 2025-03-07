#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import concurrent.futures
from asyncio.futures import Future
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Type, Union

import tornado
from tornado import httputil
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketHandler

from ads.aqua import logger
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.evaluation_ws_msg_handler import AquaEvaluationWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    AquaWsError,
    BaseRequest,
    BaseResponse,
    ErrorResponse,
    RequestResponseType,
)

MAX_WORKERS = 20


def get_aqua_internal_error_response(message_id: str) -> ErrorResponse:
    error = AquaWsError(
        status="500",
        message="Internal Server Error",
        service_payload={},
        reason="",
    )
    return ErrorResponse(
        message_id=message_id,
        kind=RequestResponseType.Error,
        data=error,
    )


class AquaUIWebSocketHandler(WebSocketHandler):
    """Handler for Aqua Websocket."""

    _handlers_: List[Type[AquaWSMsgHandler]] = [AquaEvaluationWSMsgHandler]

    thread_pool: ThreadPoolExecutor

    future_message_map: Dict[Future, BaseRequest]
    message_type_handler_map: Dict[RequestResponseType, Type[AquaWSMsgHandler]]

    def __init__(
        self,
        application: tornado.web.Application,
        request: httputil.HTTPServerRequest,
        **kwargs,
    ):
        # Create a mapping of message type to handler and check for duplicates
        self.future_message_map = {}
        self.message_type_handler_map = {}
        for handler in self._handlers_:
            for message_type in handler.get_message_types():
                if message_type in self.message_type_handler_map:
                    raise ValueError(
                        f"Duplicate message type {message_type} in AQUA websocket handlers."
                    )
                else:
                    self.message_type_handler_map[message_type] = handler

        super().__init__(application, request, **kwargs)

    def open(self, *args, **kwargs):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info("AQUA WebSocket opened")

    def on_message(self, message: Union[str, bytes]):
        try:
            request = BaseRequest.from_json(message, ignore_unknown=True)
        except Exception as e:
            logger.error(
                f"Unable to parse WebSocket message {message}\nWith exception: {str(e)}"
            )
            raise e
        # Find the handler for the message type.
        # Each handler is responsible for some specific message types
        handler = self.message_type_handler_map.get(request.kind, None)
        if handler is None:
            self.write_message(
                get_aqua_internal_error_response(request.message_id).to_json()
            )
            raise ValueError(f"No handler found for message type {request.kind}")
        else:
            message_handler = handler(message)
            future: Future = self.thread_pool.submit(message_handler.process)
            self.future_message_map[future] = request
            future.add_done_callback(self.on_message_processed)

    def on_message_processed(self, future: concurrent.futures.Future):
        """Callback function to handle the response from the various AquaWSMsgHandlers."""
        try:
            response: BaseResponse = future.result()

        # Any exception coming here is an unhandled exception in the handler. We should log it and return an internal server error.
        # In non WebSocket scenarios this would be handled by the tornado webserver
        except Exception as e:
            logger.error(
                f"Unable to handle WebSocket message {self.future_message_map[future]}\nWith exception: {str(e)}"
            )
            response: BaseResponse = get_aqua_internal_error_response(
                self.future_message_map[future].message_id
            )
            raise e
        finally:
            self.future_message_map.pop(future)
            # Send the response back to the client on the event thread
            IOLoop.current().run_sync(lambda: self.write_message(response.to_json()))

    def on_close(self) -> None:
        self.thread_pool.shutdown()
        logger.info("AQUA WebSocket closed")


__handlers__ = [("ws?([^/]*)", AquaUIWebSocketHandler)]
