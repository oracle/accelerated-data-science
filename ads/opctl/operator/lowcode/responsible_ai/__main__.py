#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import sys
from typing import List

import yaml

from ads.opctl import logger
from ads.opctl.operator.common.const import ENV_OPERATOR_ARGS
from ads.opctl.operator.common.utils import _parse_input_args
from .operator import operate, verify

from .__init__ import __name__ as MODULE


def main(raw_args: List[str]):
    """The entry point of the forecasting the operator."""
    args, _ = _parse_input_args(raw_args)
    if not args.file and not args.spec and not os.environ.get(ENV_OPERATOR_ARGS):
        logger.info(
            "Please specify -f[--file] or -s[--spec] or "
            f"pass operator's arguments via {ENV_OPERATOR_ARGS} environment variable."
        )
        return

    logger.info("-" * 100)
    logger.info(f"{'Running' if not args.verify else 'Verifying'} operator: {MODULE}")

    # if spec provided as input string, then convert the string into YAML
    operator_spec_str = args.spec or os.environ.get(ENV_OPERATOR_ARGS)

    operator_config = yaml.safe_load(operator_spec_str)

    # run operator
    if args.verify:
        # TODO: Add verify operator logic here
        print(operator_config)
    else:
        # TODO: Add run operator logic here
        
        print(operate(operator_config))


if __name__ == "__main__":
    main(sys.argv[1:])
