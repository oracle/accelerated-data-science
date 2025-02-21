#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from tqdm import tqdm


class HFModelProgressTracker(tqdm):
    def __init__(self, *args, **kwargs):
        """
        A custom tqdm class that calls `callback` each time progress is updated.

        """
        super().__init__(*args, **kwargs)

    def callback(self, *args, **kwargs):
        pass

    def update(self, n=1):
        # Perform the standard progress update
        super().update(n)
        # Invoke the callback with the current progress value (self.n)
        self.callback({"status": f"{self.n} of {self.total} files downloaded"})
