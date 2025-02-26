#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from tqdm import tqdm

from ads.aqua.common.task_status import TaskStatus, TaskStatusEnum


class HFModelProgressTracker(tqdm):
    hooks = []

    def __init__(self, *args, **kwargs):
        """
        A custom tqdm class that calls `callback` each time progress is updated.

        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def register_hooks(hook):
        HFModelProgressTracker.hooks.append(hook)

    def update(self, n=1):
        # Perform the standard progress update
        super().update(n)
        # Invoke the callback with the current progress value (self.n)
        for hook in HFModelProgressTracker.hooks:
            hook(
                TaskStatus(
                    state=TaskStatusEnum.MODEL_DOWNLOAD_INPROGRESS,
                    message=f"{self.n} of {self.total} files downloaded",
                )
            )

    def close(self):
        for hook in HFModelProgressTracker.hooks:
            hook(
                TaskStatus(
                    state=TaskStatusEnum.MODEL_DOWNLOAD_INPROGRESS,
                    message=f"{self.n} of {self.total} files downloaded",
                )
            )
        super().close()
