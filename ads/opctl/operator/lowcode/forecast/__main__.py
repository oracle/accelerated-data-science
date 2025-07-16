#!/usr/bin/env python

# Copyright (c) 2023, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import json
import os
import sys
from typing import Dict, List

import pandas as pd
import yaml

from ads.opctl import logger
from ads.opctl.operator.common.const import ENV_OPERATOR_ARGS
from ads.opctl.operator.common.utils import _parse_input_args

from .const import AUTO_SELECT_SERIES
from .model.forecast_datasets import ForecastDatasets, ForecastResults
from .operator_config import ForecastOperatorConfig
from .whatifserve import ModelDeploymentManager


def operate(operator_config: ForecastOperatorConfig) -> ForecastResults:
    """Runs the forecasting operator."""
    from .model.factory import ForecastOperatorModelFactory

    datasets = ForecastDatasets(operator_config)
    model = ForecastOperatorModelFactory.get_model(operator_config, datasets)

    if operator_config.spec.model == AUTO_SELECT_SERIES and hasattr(
        operator_config.spec, "meta_features"
    ):
        # For AUTO_SELECT_SERIES, handle each series with its specific model
        meta_features = operator_config.spec.meta_features
        results = ForecastResults()
        sub_results_list = []

        # Group the data by selected model
        for model_name in meta_features["selected_model"].unique():
            # Get series that use this model
            series_groups = meta_features[meta_features["selected_model"] == model_name]

            # Create a sub-config for this model
            sub_config = copy.deepcopy(operator_config)
            sub_config.spec.model = model_name

            # Create sub-datasets for these series
            sub_datasets = ForecastDatasets(
                operator_config,
                subset=series_groups[operator_config.spec.target_category_columns]
                .values.flatten()
                .tolist(),
            )

            # Get and run the appropriate model
            sub_model = ForecastOperatorModelFactory.get_model(sub_config, sub_datasets)
            sub_result_df, sub_elapsed_time = sub_model.build_model()
            sub_results = sub_model.generate_report(
                result_df=sub_result_df,
                elapsed_time=sub_elapsed_time,
                save_sub_reports=True,
            )
            sub_results_list.append(sub_results)

            # results_df = pd.concat([results_df, sub_result_df], ignore_index=True, axis=0)
            # elapsed_time += sub_elapsed_time
        # Merge all sub_results into a single ForecastResults object
        if sub_results_list:
            results = sub_results_list[0]
            for sub_result in sub_results_list[1:]:
                results.merge(sub_result)
        else:
            results = None

    else:
        # For other cases, use the single selected model
        results = model.generate_report()
    # saving to model catalog
    spec = operator_config.spec
    if spec.what_if_analysis and datasets.additional_data:
        mdm = ModelDeploymentManager(spec, datasets.additional_data)
        mdm.save_to_catalog()
        if spec.what_if_analysis.model_deployment:
            mdm.create_deployment()
        mdm.save_deployment_info()
    return results


def verify(spec: Dict, **kwargs: Dict) -> bool:
    """Verifies the forecasting operator config."""
    operator = ForecastOperatorConfig.from_dict(spec)
    msg_header = (
        f"{'*' * 30} The operator config has been successfully verified {'*' * 30}"
    )
    print(msg_header)
    print(operator.to_yaml())
    print("*" * len(msg_header))


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
    logger.info(f"{'Running' if not args.verify else 'Verifying'} the operator...")

    # if spec provided as input string, then convert the string into YAML
    yaml_string = ""
    if args.spec or os.environ.get(ENV_OPERATOR_ARGS):
        operator_spec_str = args.spec or os.environ.get(ENV_OPERATOR_ARGS)
        try:
            yaml_string = yaml.safe_dump(json.loads(operator_spec_str))
        except json.JSONDecodeError:
            yaml_string = yaml.safe_dump(yaml.safe_load(operator_spec_str))
        except:
            yaml_string = operator_spec_str

    operator_config = ForecastOperatorConfig.from_yaml(
        uri=args.file,
        yaml_string=yaml_string,
    )

    # run operator
    if args.verify:
        verify(operator_config)
    else:
        operate(operator_config)


if __name__ == "__main__":
    main(sys.argv[1:])
