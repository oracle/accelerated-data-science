#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

from ads.common.decorator.deprecate import deprecated
from multiprocessing.pool import ThreadPool

try:
    import dask
    from dask.distributed import Client
except ImportError as e:
    raise ModuleNotFoundError("Install dask to use the MLRuntime class.") from e

_default_dask_threadpool_size = 50
_client = None
_storage = None


class MLRuntime(object):
    @deprecated("2.5.2")
    def __init__(self):
        dask_threadpool_size = (
            os.environ.get("DASK_THREADPOOL_SIZE") or _default_dask_threadpool_size
        )
        dask.config.set(pool=ThreadPool(int(dask_threadpool_size)))
        if "DASK_SCHEDULER" in os.environ:
            self.dask_client = Client(address=os.environ["DASK_SCHEDULER"])
        elif os.environ.get("DASK_LOCAL_CLIENT", "False").lower() == "true":
            self.dask_client = Client(
                processes=False,
                local_directory="/tmp",  # use directory with write permits
            )
        else:
            # first let's see if the default client is already running, if it is
            # connect to it, else connect a new Dask scheduler
            # localhost:8786 for dask scheduler
            try:
                self.dask_client = Client("127.0.0.1:8786", timeout=1)
            except:
                self.dask_client = Client(
                    processes=False,  # use threads, avoid for inproc cluster IPC overhead
                    name="Oracle Data Science Local Dask",
                    heartbeat_interval=60000,  # one mimnute heartbeats
                    local_directory="/tmp",  # use directory with write permits
                )

    def get_compute_accelerator(self):
        return self.dask_client


def init():
    global _client

    # ensures the client is created only once
    _client = MLRuntime().get_compute_accelerator()


def compute_accelerator():
    global _client
    if _client is None:
        init()
    return _client
