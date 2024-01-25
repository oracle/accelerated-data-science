#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class ModelDeploymentNotFoundError(Exception):
    """Exception raised when the model deployment with the given ID cannot be found."""

    def __init__(self, model_id: str):
        super().__init__(
            f"The model deployment with ID: `{model_id}` cannot be found. "
            "Please check if the model deployment exists."
        )


class SessionNotFoundError(Exception):
    """Exception raised when the session with the given ID cannot be found."""

    def __init__(self, search_id: str):
        super().__init__(
            f"The session with provided ID: `{search_id}` cannot be found. "
            "Please ensure that the session with given ID exists."
        )


class ThreadNotFoundError(Exception):
    """Exception raised when the thread with the given ID cannot be found."""

    def __init__(self, thread_id: str):
        super().__init__(
            f"The thread with provided ID: `{thread_id}` cannot be found. "
            "Please ensure that the thread with given ID exists."
        )
