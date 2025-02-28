#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Callable, List, Union

from tqdm import tqdm

from ads.aqua.common.task_status import TaskStatus, TaskStatusEnum


class HFModelProgressTracker(tqdm):
    """snapshot_download method from huggingface_hub library is used to download the models. This class provides a way to register for callbacks as the downloads of different files are complete."""

    hooks = []

    def __init__(self, *args, **kwargs):
        """
        A custom tqdm class that calls `callback` each time progress is updated.

        """
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        # Perform the standard progress update
        super().update(n)
        # Invoke the callback with the current progress value (self.n)
        for hook in self.hooks:
            hook(
                TaskStatus(
                    state=TaskStatusEnum.MODEL_DOWNLOAD_INPROGRESS,
                    message=f"{self.n} of {self.total} files downloaded",
                )
            )

    def close(self):
        for hook in self.hooks:
            hook(
                TaskStatus(
                    state=TaskStatusEnum.MODEL_DOWNLOAD_INPROGRESS,
                    message=f"{self.n} of {self.total} files downloaded",
                )
            )
        super().close()


def prepare_progress_tracker_with_callback(
    hook: Union[Callable, List[Callable]],
) -> "HFModelProgressTrackerWithHook":  # type: ignore  # noqa: F821
    """Provide a list of callables or single callable to be invoked upon download progress. snapshot_download only allows to pass in class, does not allow for tqdm_kwargs supported by thread_map.
    This class provides a thread safe way to use hooks"""

    class HFModelProgressTrackerWithHook(HFModelProgressTracker):
        hooks = hook if isinstance(hook, list) else [hook]

    return HFModelProgressTrackerWithHook
