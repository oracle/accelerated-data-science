#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock

from tqdm.contrib.concurrent import thread_map

from ads.aqua.model.utils import prepare_progress_tracker_with_callback


def test_custom_tqdm_thread_map():
    def process(item):
        import time

        time.sleep(0.1)
        return item

    items = list(range(0, 10))
    callback = MagicMock()

    clz = prepare_progress_tracker_with_callback(callback)
    thread_map(
        process,
        items,
        desc=f"Fetching {len(items)} items",
        tqdm_class=clz,
        max_workers=3,
    )
    callback.assert_called()


def test_custom_tqdm():
    callback = MagicMock()
    clz = prepare_progress_tracker_with_callback(callback)
    with clz(range(10), desc="Processing") as bar:
        for _ in bar:
            # Simulate work
            import time

            time.sleep(0.01)
    callback.assert_called()


if __name__ == "__main__":
    test_custom_tqdm_thread_map()
