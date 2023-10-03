#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class TestCommonUtils:
    """Tests common utils in the operator_loader module."""

    def test_operator_info(self):
        pass

    def test_operator_info_list(self):
        pass


class TestOperatorInfo:
    """Tests the class representing brief information about the operator."""

    def test_init(self):
        pass

    def test_conda_prefix(self):
        pass

    def test_from_yaml(self):
        pass


class TestOperatorLoader:
    """Tests operator loader class"""

    def test_init(self):
        pass

    def test_from_uri_success(self):
        pass

    def test_from_uri_fail(self):
        pass


class TestServiceOperatorLoader:
    """Tests class to load a service operator."""

    def test_compatible(self):
        pass

    def test_load(self):
        pass


class TestLocalOperatorLoader:
    """Tests class to load a local operator."""

    def test_compatible(self):
        pass

    def test_load(self):
        pass


class TestRemoteOperatorLoader:
    """Tests class to load a remote operator."""

    def test_compatible(self):
        pass

    def test_load(self):
        pass

    def test_cleanup(self):
        pass


class TestGitOperatorLoader:
    """Tests class to load an operator from a GIT repository."""

    def test_compatible(self):
        pass

    def test_load(self):
        pass

    def test_cleanup(self):
        pass
