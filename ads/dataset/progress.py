#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time
from abc import abstractmethod

from oci._vendor import six
from tqdm.auto import tqdm


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


class TqdmProgressBar(ProgressBar):
    def __enter__(self):
        return self

    def __init__(self, max_progress=100, description="Running", verbose=False):
        self.max_progress = max_progress
        self.progress_bar = tqdm(
            total=max_progress, desc="loop1", leave=False, mininterval=0
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
            self.progress_bar.set_description(description, refresh=True)

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
