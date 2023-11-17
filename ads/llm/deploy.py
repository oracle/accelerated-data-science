#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import importlib.util
import os
import tempfile
import yaml
from datetime import datetime
from typing import Any

import fsspec
import yaml
from jinja2 import Environment, PackageLoader

from ads.model.artifact import ADS_VERSION, SCORE_VERSION
from ads.model.generic_model import GenericModel
from ads.llm.serialize import dump, load


class ChainDeployment(GenericModel):
    def __init__(self, chain, **kwargs):
        self.chain = chain
        super().__init__(**kwargs)

    def prepare(self, **kwargs) -> GenericModel:
        chain_yaml_uri = os.path.join(self.artifact_dir, "chain.yaml")
        with open(chain_yaml_uri, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(dump(self.chain)))

        if "score_py_uri" not in kwargs:
            score_py_uri = os.path.join(tempfile.mkdtemp(), "score.py")
            env = Environment(loader=PackageLoader("ads", "llm/templates"))
            score_template = env.get_template("score_chain.jinja2")
            time_suffix = datetime.today().strftime("%Y%m%d_%H%M%S")

            context = {
                "SCORE_VERSION": SCORE_VERSION,
                "ADS_VERSION": ADS_VERSION,
                "time_created": time_suffix,
            }
            with fsspec.open(score_py_uri, "w") as f:
                f.write(score_template.render(context))
            kwargs["score_py_uri"] = score_py_uri

        return super().prepare(**kwargs)

    @classmethod
    def load_chain(cls, yaml_uri: str) -> Any:
        chain_dict = {}
        with open(yaml_uri, "r", encoding="utf-8") as file:
            chain_dict = yaml.safe_load(file)

        return load(chain_dict)
