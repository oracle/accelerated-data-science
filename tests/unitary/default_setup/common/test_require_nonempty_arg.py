#!/usr/bin/env python

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest

from ads.common.decorator.require_nonempty_arg import require_nonempty_arg


class TestRequireNonEmptyArg:
    def test_require_nonempty_single_arg_empty(self):
        # Test that the decorator raises a ValueError when a
        # single required argument is empty
        @require_nonempty_arg("arg1")
        def func(arg1):
            return True

        with pytest.raises(ValueError):
            func("")

    def test_require_nonempty_single_arg_non_empty(self):
        # Test that the decorator does not raise an exception when a
        # single required argument is not empty
        @require_nonempty_arg("arg1")
        def func(arg1):
            return True

        assert func("non-empty") == True

    def test_require_nonempty_multiple_args_some_empty(self):
        # Test that the decorator raises a ValueError when at least
        # one of multiple required arguments is empty
        @require_nonempty_arg(["arg1", "arg2"])
        def func(arg1, arg2):
            return True

        assert func("non-empty", None) == True

    def test_require_nonempty_multiple_args_none_empty(self):
        # Test that the decorator does not raise an exception when none of
        # the multiple required arguments are empty
        @require_nonempty_arg(["arg1", "arg2"])
        def func(arg1, arg2):
            return True

        assert func("non-empty", "non-empty") == True

    def test_custom_error_message(self):
        # Test that a custom error message is correctly used
        # when a required argument is empty
        custom_message = "Custom error message"

        @require_nonempty_arg("arg1", error_msg=custom_message)
        def func(arg1):
            return True

        with pytest.raises(ValueError) as exc_info:
            func("")

        assert str(exc_info.value) == custom_message
        assert str(exc_info.value) == custom_message
