#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from tornado.websocket import WebSocketHandler

from ads.aqua.extension.evaluation_ws_msg_handler import AquaEvaluationWSMsgHandler
from ads.aqua.extension.ui_websocket_handler import AquaUIWebSocketHandler


class TestAquaUIWebSocketHandler(unittest.TestCase):
    @patch.object(WebSocketHandler, "__init__")
    def setUp(self, webSocketInitMock) -> None:
        webSocketInitMock.return_value = None
        self.web_socket_handler = AquaUIWebSocketHandler(MagicMock(), MagicMock())

    def test_throws_error_on_duplicate_msg_handlers(self):
        """Test that an error is thrown when duplicate message handlers are added."""
        with self.assertRaises(ValueError):
            AquaUIWebSocketHandler._handlers_.append(AquaEvaluationWSMsgHandler)
            AquaUIWebSocketHandler(MagicMock(), MagicMock())
        AquaUIWebSocketHandler._handlers_.pop()

    def test_throws_error_on_bad_request(self):
        """Test that an error is thrown when a bad request is made."""
        with self.assertRaises(ValueError):
            self.web_socket_handler.on_message("test")

    @patch.object(AquaUIWebSocketHandler, "write_message")
    def test_throws_error_on_unexpected_kind(self, write_message_mock: MagicMock):
        """Test that an error is thrown when an unexpected kind is received."""
        write_message_mock.return_value = None
        with self.assertRaises(ValueError):
            self.web_socket_handler.on_message(
                '{"message_id": "test", "kind": "test", "data": {}}'
            )
        assert write_message_mock.called

    @patch.object(AquaUIWebSocketHandler, "write_message")
    @patch("ads.aqua.extension.ui_websocket_handler.IOLoop")
    def test_throws_internal_error_on_future_error(
        self, ioloop_mock: MagicMock, write_message_mock: MagicMock
    ):
        future = Future()
        future.set_exception(ValueError())
        self.web_socket_handler.future_message_map[future] = MagicMock()
        with self.assertRaises(ValueError):
            self.web_socket_handler.on_message_processed(future)
        assert ioloop_mock.current().run_sync.called
