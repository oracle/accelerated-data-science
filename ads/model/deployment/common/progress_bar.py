#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


# Standard lib
import time

# Third party
from abc import abstractmethod
from tqdm import tqdm_notebook
from oci._vendor import six


class ProgressBar(object):
    """ProgressBar is an abstract class for creating progress bars.

    Methods
    -------
    __enter__()
        runtime context entry method
    update(description)
        abstract method for update
    __exit__(exc_type, exc_val, exc_tb)
        runtime context exit method

    """

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def update(self, description):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TqdmProgressBar(ProgressBar):
    """TqdmProgressBar represents a progress bar for notebooks. It inherits from the
    abstract class ProgressBar

    Attributes
    ----------
    max_progress (int)
        The maximum value for the progress bar
    decription (str)
        The progres bar's description
    progress_bar (tqdm_notebook)
        Notebook widget
    verbose (bool)
        verbosity flag

    Methods
    -------
    __enter__()
        runtime context entry method
    __init__(max_progress, description, verbose)
        init method for class
    update(description=None)
        updates progress bar
    __exit__(exc_type, exc_val, exc_tb)
        runtime context exit method
    """

    def __enter__(self):
        """__enter__ runs when the context starts. It returns the class object"""
        return self

    def __init__(self, max_progress=100, description="Running", verbose=False):
        """Class initialization function

        Args:
            max_progress (int, optional): the maximum progress bar value (defaults to 100)
            description (str, optional): the progress bar description (defaults to "Running")
            verbose (bool, optional): verbosity flag (defaults to False)
        """

        self.max_progress = max_progress
        self.progress_bar = tqdm_notebook(
            range(max_progress), desc="loop1", mininterval=0, leave=False
        )
        self.progress_bar.set_description(description)
        self.verbose = verbose
        if self.verbose:
            self.start_time = time.time()
            self.description = description

    def update(self, description=None):
        """update updates the progress bar

        Args:
            description (str, optional): progress bar description

        Returns:
            Nothing
        """

        if self.verbose and description is not None:
            print(
                "%s:%ss"
                % (
                    self.description,
                    str.format("{0:.3f}", time.time() - self.start_time),
                )
            )
            self.description = (
                description if description is not None else self.description
            )
            self.start_time = time.time()
        self.progress_bar.update(1)
        if description is not None:
            self.progress_bar.set_description(description)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__ runs when the context closes. On success, i.e. all arguments are None
        it sets the description and closes the context. Otherwise it reraises an exception
        thrown in the context

        Args:
            exc_type (Exception): exception type
            exc_val (Exception value): exception value
            exc_tb (Traceback): exception traceback

        Returns:
            Nothing
        """

        if exc_tb is None:
            self.progress_bar.set_description("Done")
            self.progress_bar.close()
            return
        six.reraise(exc_type, exc_val, exc_tb)


class DummyProgressBar(ProgressBar):
    """DummyProgressBar is represents a progress bar for non-notebook environments. It inherits
    from the abstract class ProgressBar. It allows use of the same contextlib enter and exit methods
    """

    def __enter__(self):
        return self

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
