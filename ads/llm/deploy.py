#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
import tempfile
from datetime import datetime
from typing import Any

import yaml
from jinja2 import Environment, PackageLoader

from ads.model.artifact import ADS_VERSION, SCORE_VERSION
from ads.model.generic_model import GenericModel
from ads.llm.serialize import dump, load_from_yaml


class ChainDeployment(GenericModel):
    def __init__(self, chain, **kwargs):
        self.chain = chain
        super().__init__(**kwargs)

    def prepare(self, **kwargs) -> GenericModel:
        """Prepares the model artifact."""
        chain_yaml_uri = os.path.join(self.artifact_dir, "chain.yaml")
        with open(chain_yaml_uri, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(dump(self.chain)))

        try:
            score_py = None
            if "score_py_uri" not in kwargs:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="score.py", delete=False
                ) as score_py:
                    env = Environment(loader=PackageLoader("ads", "llm/templates"))
                    score_template = env.get_template("score_chain.jinja2")
                    time_suffix = datetime.today().strftime("%Y%m%d_%H%M%S")

                    context = {
                        "SCORE_VERSION": SCORE_VERSION,
                        "ADS_VERSION": ADS_VERSION,
                        "time_created": time_suffix,
                    }
                    score_py.write(score_template.render(context))

                kwargs["score_py_uri"] = score_py.name
            return super().prepare(**kwargs)
        finally:
            if score_py:
                os.unlink(score_py.name)

    @classmethod
    def load_chain(cls, yaml_uri: str, **kwargs) -> Any:
        return load_from_yaml(yaml_uri, **kwargs)
