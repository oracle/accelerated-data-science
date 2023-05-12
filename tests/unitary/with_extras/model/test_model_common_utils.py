#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from ads.model.common.utils import _extract_locals


class TestUtils:
    @pytest.mark.parametrize(
        "input_data, expected_result",
        [
            (
                {"locals": {"a": "a", "b": "b", "c": None}, "filter_out_nulls": False},
                {"a": "a", "b": "b", "c": None},
            ),
            ({"locals": {"a": "a", "b": "b", "c": None}}, {"a": "a", "b": "b"}),
            (
                {
                    "locals": {
                        "a": "a",
                        "b": "b",
                        "c": None,
                        "kwargs": {"d": "d", "e": None},
                    }
                },
                {"a": "a", "b": "b", "d": "d"},
            ),
        ],
    )
    def test__extract_locals(self, input_data, expected_result):
        """Tests extracting arguments from local variables."""
        test_result = _extract_locals(**input_data)
        assert test_result == expected_result
