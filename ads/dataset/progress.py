#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time
from abc import abstractmethod

from oci._vendor import six
from tqdm import tqdm_notebook
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class ProgressBar(object):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def update(self, description):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class IpythonProgressBar(ProgressBar):
    @runtime_dependency(module="ipywidgets", install_from=OptionalDependency.NOTEBOOK)
    def __init__(self, max_progress=100, description="Running", verbose=False):
        self.max_progress = max_progress

        from ads.common import logger

        from ipywidgets import Label, IntProgress

        self.progress_label = Label(description)
        self.progress_bar = IntProgress(min=0, max=max_progress)
        self.verbose = verbose
        if self.verbose:
            self.start_time = time.time()

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def __enter__(self):
        from IPython.core.display import display

        display(self.progress_label)
        display(self.progress_bar)
        return self

    def update(self, description=None):
        """
        Updates the progress bar
        """
        if self.verbose:
            print(
                str.format("{0:.3f}", time.time() - self.start_time),
                self.progress_label.value,
            )
            self.start_time = time.time()
        self.progress_bar.value += 1
        if description is not None:
            self.progress_label.value = description

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.progress_label.value = "Done"
            self.progress_bar.value = self.max_progress
            return
        raise exc_type(exc_val).with_traceback(exc_tb)


class TqdmProgressBar(ProgressBar):
    def __enter__(self):
        return self

    @runtime_dependency(module="ipywidgets", install_from=OptionalDependency.NOTEBOOK)
    def __init__(self, max_progress=100, description="Running", verbose=False):
        self.max_progress = max_progress

        from ads.common import logger

        self.progress_bar = tqdm_notebook(
            range(max_progress), desc="loop1", mininterval=0, leave=False
        )
        self.progress_bar.set_description(description)
        self.verbose = verbose
        if self.verbose:
            self.start_time = time.time()
            self.description = description

    def update(self, description=None):
        """
        Updates the progress bar
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
        if exc_tb is None:
            self.progress_bar.set_description("Done")
            self.progress_bar.close()
            return
        six.reraise(exc_type, exc_val, exc_tb)


class DummyProgressBar(ProgressBar):
    def __enter__(self):
        return self

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        """
        Updates the progress bar
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
