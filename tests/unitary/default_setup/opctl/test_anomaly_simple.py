#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import subprocess
import pandas as pd
import pytest
from time import sleep
from copy import deepcopy
import tempfile
import os


TEMPLATE_YAML = {
    "kind": "operator",
    "type": "anomaly",
    "version": "v1",
    "spec": {
        "input_data": {
            "url": "https://raw.githubusercontent.com/facebook/prophet/c00f6a2d72229faa6acee8292bc01e14f16f599c/examples/example_retail_sales.csv",
        },
        "output_directory": {
            "url": "results",
        },
        "datetime_column": {
            "name": "ds",
        },
    },
}


def test_load_datasets():
    for model in [
        "automlx"
    ]:  # ["automlx", "prophet", "neuralprophet", "autots", "arima", "auto"]
        with tempfile.TemporaryDirectory() as tmpdirname:
            forecast_yaml_filename = f"{tmpdirname}/anomaly.yaml"
            output_dirname = f"{tmpdirname}/results"

            yaml_i = deepcopy(TEMPLATE_YAML)
            yaml_i["spec"]["model"] = model
            yaml_i["spec"]["output_directory"]["url"] = output_dirname

            with open(f"{tmpdirname}/anomaly.yaml", "w") as f:
                f.write(yaml.dump(yaml_i))
            sleep(0.5)
            subprocess.run(
                f"ads operator run -f {forecast_yaml_filename} --debug", shell=True
            )
            sleep(0.1)
            subprocess.run(f"ls -a {output_dirname}/", shell=True)

            # train_metrics = pd.read_csv(f"{output_dirname}/metrics.csv")
            # print(train_metrics)
            # fcst = pd.read_csv(f"{output_dirname}/anomaly.csv")
            # print(fcst)
            assert os.path.exists(
                f"{output_dirname}/report.html"
            ), "Report not generated."

        # if TEMPLATE_YAML["spec"]["generate_explanations"]:
        # glb_expl = pd.read_csv(f"{tmpdirname}/results/global_explanation.csv")
        # print(glb_expl)
        # loc_expl = pd.read_csv(f"{tmpdirname}/results/local_explanation.csv")
        # print(loc_expl)
