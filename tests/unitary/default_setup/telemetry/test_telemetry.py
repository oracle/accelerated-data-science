#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import patch

import pytest

from ads.telemetry import Telemetry


class TestTelemetry:
    """Tests the Telemetry.
    Class to capture telemetry sequence into the environment variable.
    """

    def setup_method(self):
        self.telemetry = Telemetry(name="test.api")

    @patch.dict(os.environ, {}, clear=True)
    def test_init(self):
        """Ensures initializing Telemetry passes."""
        self.telemetry = Telemetry("test.api")
        assert self.telemetry.name == "test.api"
        assert self.telemetry.environ_variable in os.environ
        assert os.environ[self.telemetry.environ_variable] == ""

    @patch.dict(os.environ, {}, clear=True)
    def test_add(self):
        """Tests adding the new value to the telemetry."""
        self.telemetry.begin()
        self.telemetry.add("key=value").add("new_key=new_value")
        assert (
            os.environ[self.telemetry.environ_variable]
            == "test.api&key=value&new_key=new_value"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_begin(self):
        """Tests cleaning the value of the associated environment variable."""
        self.telemetry.begin("key=value")
        assert os.environ[self.telemetry.environ_variable] == "test.api&key=value"

    @patch.dict(os.environ, {}, clear=True)
    def test_clean(self):
        """Ensures that telemetry associated environment variable can be cleaned."""
        self.telemetry.begin()
        self.telemetry.add("key=value").add("new_key=new_value")
        assert (
            os.environ[self.telemetry.environ_variable]
            == "test.api&key=value&new_key=new_value"
        )
        self.telemetry.clean()
        assert os.environ[self.telemetry.environ_variable] == ""

    @patch.dict(os.environ, {"EXTRA_USER_AGENT_INFO": "some_existing_value"}, clear=True)
    def test_restore(self):
        """Ensures that telemetry associated environment variable can be restored to the original value."""
        telemetry = Telemetry(name="test.api")
        telemetry.begin()
        telemetry.add("key=value").add("new_key=new_value")
        assert (
            os.environ[telemetry.environ_variable]
            == "test.api&key=value&new_key=new_value"
        )
        telemetry.restore()
        assert os.environ[telemetry.environ_variable] == "some_existing_value"

    @pytest.mark.parametrize(
        "NAME,INPUT_DATA,EXPECTED_RESULT",
        [
            ("test.api", "key=va~!@#$%^*()_+lue", "key=va____________lue"),
            ("test.api", "key=va     lue", "key=va_lue"),
            ("", "key=va123***lue", "key=va123___lue"),
            ("", "", ""),
        ],
    )
    @patch.dict(os.environ, {}, clear=True)
    def test__prepare(self, NAME, INPUT_DATA, EXPECTED_RESULT):
        """Tests replacing special characters in the telemetry input value."""
        telemetry = Telemetry(name=NAME)
        telemetry.begin(INPUT_DATA)
        expected_result = f"{NAME}&{EXPECTED_RESULT}" if NAME else EXPECTED_RESULT
        assert os.environ[telemetry.environ_variable] == expected_result

