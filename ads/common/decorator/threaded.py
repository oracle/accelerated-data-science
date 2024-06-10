#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import concurrent.futures
import functools
import logging
from typing import Optional

from git import Optional

from ads.config import THREADED_DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

# Create a global thread pool with a maximum of 10 threads
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)


class TimeoutError(Exception):
    """
    Custom exception to be raised when a function times out.

    Attributes
    ----------
    message : str
        The error message describing what went wrong.

    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(
        self,
        func_name: str,
        timeout: int,
        message: Optional[str] = "The operation could not be completed in time.",
    ):
        super().__init__(
            f"{message} The function '{func_name}' exceeded the timeout of {timeout} seconds."
        )


def threaded(timeout: Optional[int] = THREADED_DEFAULT_TIMEOUT):
    """
    Decorator to run a function in a separate thread using a global thread pool.

    Parameters
    ----------
    timeout (int, optional)
        The maximum time in seconds to wait for the function to complete.
        If the function does not complete within this time, "timeout" is returned.

    Returns
    -------
    function: The wrapped function that will run in a separate thread with the specified timeout.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function to submit the decorated function to the thread pool and handle timeout.

            Parameters
            ----------
                *args: Positional arguments to pass to the decorated function.
                **kwargs: Keyword arguments to pass to the decorated function.

            Returns
            -------
            Any: The result of the decorated function if it completes within the timeout.

            Raise
            -----
            TimeoutError
                In case of the function exceeded the timeout.
            """
            future = thread_pool.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as ex:
                logger.debug(
                    f"The function '{func.__name__}' "
                    f"exceeded the timeout of {timeout} seconds. "
                    f"{ex}"
                )
                raise TimeoutError(func.__name__, timeout)

        return wrapper

    return decorator
