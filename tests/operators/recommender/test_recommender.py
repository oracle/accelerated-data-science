#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import subprocess
import tempfile
from time import sleep

import pytest
import yaml

DATASET_PREFIX = f"{os.path.dirname(os.path.abspath(__file__))}/../data/recommendation/"


@pytest.mark.skip()
def test_recommender():
    user_file = f"{DATASET_PREFIX}users.csv"
    item_file = f"{DATASET_PREFIX}items.csv"
    interation_file = f"{DATASET_PREFIX}interactions.csv"

    yaml_params = {
        "kind": "operator",
        "type": "recommender",
        "version": "v1",
        "spec": {
            "user_data": {
                "url": user_file,
            },
            "item_data": {
                "url": item_file,
            },
            "interactions_data": {
                "url": interation_file,
            },
            "output_directory": {
                "url": "results",
            },
            "top_k": 4,
            "model_name": "svd",
            "user_column": "user_id",
            "item_column": "movie_id",
            "interaction_column": "rating",
            "recommendations_filename": "recommendations.csv",
            "generate_report": True,
            "report_filename": "report.html",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        recommender_yaml_filename = f"{tmpdirname}/recommender.yaml"
        output_dirname = f"{tmpdirname}/results"
        yaml_params["spec"]["output_directory"]["url"] = output_dirname
        with open(recommender_yaml_filename, "w") as f:
            f.write(yaml.dump(yaml_params))
        sleep(0.5)
        subprocess.run(
            f"ads operator run -f {recommender_yaml_filename} --debug", shell=True
        )
        sleep(0.1)
        assert os.path.exists(f"{output_dirname}/report.html"), "Report not generated."
