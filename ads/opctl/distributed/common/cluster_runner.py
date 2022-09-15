#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from time import sleep, time_ns
from ads.opctl.distributed.common.cluster_provider_factory import ClusterProviderFactory
import traceback


class ClusterRunner:
    def __init__(self, cluster_provider=None, cluster_key=None):
        self.cluster_key = cluster_key or os.environ.get("OCI__CLUSTER_TYPE")
        self.mode = os.environ.get("OCI__MODE")
        self.ephemeral = os.environ.get("OCI__EPHEMERAL", 1)
        # life_span = os.environ.get("OCI__LIFE_SPAN")  # TODO: Ask MR how this works
        self.work_dir = os.environ.get("OCI__WORK_DIR")
        os.environ["JOB_OCID"] = os.environ.get("JOB_OCID", 'Undefined')
        os.environ["JOB_RUN_OCID"] = os.environ.get("JOB_RUN_OCID", str(time_ns()))
        self.cluster = cluster_provider or ClusterProviderFactory.get_provider(
            self.cluster_key,
            mode=self.mode,
            ephemeral=self.ephemeral,
            work_dir=self.work_dir,
        )  # life_spanlife_span=life_span
        print(f"Cluster built: {self.cluster}", flush=True)

    def run(self):
        exit_code = 0
        self.cluster.start()
        try:
            self.cluster.run_code()
            # self.cluster.code_execution_complete = True # This needs to be
            # set inside the run_code method of the implementation class.
        except Exception as e:
            print(f"Error Running the code: {e}", flush=True)
            traceback.print_exc()
            exit_code = 1
            self.cluster.execution_failed()
        while (
            not self.cluster.tearable()
        ):  # If not ephemeral, wait util it is ready for tearing down
            sleep(15)
        print("Signalling Stop!!!", flush=True)
        self.cluster.stop()  # Signal cluster tear down
        print(f"Exiting with exit code: {exit_code}", flush=True)
        self.cluster.sync(loop=False)
        exit(exit_code)


if __name__ == "__main__":
    ClusterRunner().run()
