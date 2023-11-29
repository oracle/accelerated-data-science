#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class TestOperatorCMD:
    """Tests operator commands."""

    def test_list(self):
        """Ensures that the list of the registered operators can be printed."""
        pass

    def test_info(self):
        """Ensures that the detailed information about the particular operator can be printed."""
        pass

    def test_init_success(self):
        """Ensures that a starter YAML configurations for the operator can be generated."""
        pass

    def test_init_fail(self):
        """Ensures that generating starter specification fails in case of wrong input attributes."""
        pass

    def test_build_image(self):
        """Ensures that operator's image can be successfully built."""
        pass

    def test_publish_image(self):
        """Ensures that operator's image can be successfully published."""
        pass

    def test_verify(self):
        """Ensures that operator's config can be successfully verified."""
        pass

    def test_build_conda(self):
        """Ensures that the operator's conda environment can be successfully built."""
        pass

    def test_publish_conda(self):
        """Ensures that the operator's conda environment can be successfully published."""
        pass

    def test_run(self):
        """Ensures that the operator can be run on the targeted backend."""
        pass
