#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import sys
from typing import Dict, List

import yaml

from ads.opctl import logger
from ads.opctl.operator.common.const import ENV_OPERATOR_ARGS
from ads.opctl.operator.common.utils import _parse_input_args

from .model.recommender_dataset import RecommenderDatasets
from .operator_config import RecommenderOperatorConfig
from .model.factory import RecommenderOperatorModelFactory

def operate(operator_config: RecommenderOperatorConfig) -> None:
    """Runs the recommender operator."""

    datasets = RecommenderDatasets(operator_config)
    RecommenderOperatorModelFactory.get_model(
        operator_config, datasets
    ).generate_report()


def verify(spec: Dict, **kwargs: Dict) -> bool:
    """Verifies the recommender detection operator config."""
    operator = RecommenderOperatorConfig.from_dict(spec)
    msg_header = (
        f"{'*' * 50} The operator config has been successfully verified {'*' * 50}"
    )
    print(msg_header)
    print(operator.to_yaml())
    print("*" * len(msg_header))


def main(raw_args: List[str]):
    """The entry point of the recommender the operator."""
    args, _ = _parse_input_args(raw_args)
    if not args.file and not args.spec and not os.environ.get(ENV_OPERATOR_ARGS):
        logger.info(
            "Please specify -f[--file] or -s[--spec] or "
            f"pass operator's arguments via {ENV_OPERATOR_ARGS} environment variable."
        )
        return

    logger.info("-" * 100)
    logger.info(
        f"{'Running' if not args.verify else 'Verifying'} the recommender detection operator."
    )

    yaml_string = ""
    if args.spec or os.environ.get(ENV_OPERATOR_ARGS):
        operator_spec_str = args.spec or os.environ.get(ENV_OPERATOR_ARGS)
        try:
            yaml_string = yaml.safe_dump(json.loads(operator_spec_str))
        except json.JSONDecodeError:
            yaml_string = yaml.safe_dump(yaml.safe_load(operator_spec_str))
        except:
            yaml_string = operator_spec_str

    operator_config = RecommenderOperatorConfig.from_yaml(
        uri=args.file,
        yaml_string=yaml_string,
    )

    logger.info(operator_config.to_yaml())

    # run operator
    if args.verify:
        verify(operator_config)
    else:
        operate(operator_config)


if __name__ == "__main__":
    main(sys.argv[1:])
