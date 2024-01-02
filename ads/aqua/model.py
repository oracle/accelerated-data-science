#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class AquaModel:
    """APIs for Aqua Models."""

    @classmethod
    def list(cls, compartment_id, project_id, **kwargs):
        """List Aqua models."""
        print("This command lists the AQUA models.")
        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})
        return kwargs