#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import sys

from ads.mloperator.common.utils import _parse_input_args

from .__init__ import __description__ as DESCRIPTION
from .__init__ import __name__ as MODULE
from .main import ForecastOperator, run

ENV_OPERATOR_ARGS = "ENV_OPERATOR_ARGS"


def main(raw_args):
    args, _ = _parse_input_args(raw_args)
    if not args.file and not args.spec and not os.environ.get(ENV_OPERATOR_ARGS):
        print(
            "Please specify -f[--file] or -s[--spec] or "
            f"pass operator's arguments via {ENV_OPERATOR_ARGS} environment variable."
        )
        return

    print("-" * 100)
    print(f"Running operator: {MODULE}")
    print(DESCRIPTION)

    operator = ForecastOperator.from_yaml(
        uri=args.file,
        yaml_string=args.spec or os.environ.get(ENV_OPERATOR_ARGS, ""),
    )

    print(operator.to_yaml())
    run(operator)


if __name__ == "__main__":
    main(sys.argv[1:])
