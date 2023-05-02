#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class ChangesNotCommitted(Exception):   # pragma: no cover
    def __init__(self, path):
        msg = f"""
            File(s) at {path} are either dirty or untracked.
            Please commit changes and then save the model, or set `ignore_pending_changes=True`.
        """
        super().__init__(msg)
