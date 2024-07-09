#!/usr/bin/env python

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

from .model.anomaly_dataset import AnomalyDatasets
from .operator_config import AnomalyOperatorConfig


def operate(operator_config: AnomalyOperatorConfig) -> None:
    """Runs the anomaly detection operator."""
    from .model.factory import AnomalyOperatorModelFactory

    datasets = AnomalyDatasets(operator_config.spec)
    try:
        AnomalyOperatorModelFactory.get_model(
            operator_config, datasets
        ).generate_report()
    except Exception as e:
        if operator_config.spec.model == "auto":
            logger.debug(
                f"Failed to forecast with error {e.args}. Trying again with model `autots`."
            )
            operator_config.spec.model = "autots"
            operator_config.spec.model_kwargs = {}
            datasets = AnomalyDatasets(operator_config.spec)
            try:
                AnomalyOperatorModelFactory.get_model(
                    operator_config, datasets
                ).generate_report()
            except Exception as ee:
                logger.debug(
                    f"Failed to backup forecast with error {ee.args}. Raising original error."
                )
                raise ee
        else:
            raise e


def verify(spec: Dict) -> bool:
    """Verifies the anomaly detection operator config."""
    operator = AnomalyOperatorConfig.from_dict(spec)
    msg_header = (
        f"{'*' * 50} The operator config has been successfully verified {'*' * 50}"
    )
    print(msg_header)
    print(operator.to_yaml())
    print("*" * len(msg_header))


def main(raw_args: List[str]):
    """The entry point of the anomaly the operator."""
    args, _ = _parse_input_args(raw_args)
    if not args.file and not args.spec and not os.environ.get(ENV_OPERATOR_ARGS):
        logger.info(
            "Please specify -f[--file] or -s[--spec] or "
            f"pass operator's arguments via {ENV_OPERATOR_ARGS} environment variable."
        )
        return

    logger.info("-" * 100)
    logger.info(
        f"{'Running' if not args.verify else 'Verifying'} the anomaly detection operator."
    )

    # if spec provided as input string, then convert the string into YAML
    yaml_string = ""
    if args.spec or os.environ.get(ENV_OPERATOR_ARGS):
        operator_spec_str = args.spec or os.environ.get(ENV_OPERATOR_ARGS)
        try:
            yaml_string = yaml.safe_dump(json.loads(operator_spec_str))
        except json.JSONDecodeError:
            yaml_string = yaml.safe_dump(yaml.safe_load(operator_spec_str))
        except Exception:
            yaml_string = operator_spec_str

    operator_config = AnomalyOperatorConfig.from_yaml(
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
