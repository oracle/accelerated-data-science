#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from darts import datasets as d_datasets
import yaml
import tempfile
import subprocess
import pandas as pd
import pytest
from time import sleep, time
from copy import deepcopy
from pathlib import Path
import random
import pathlib
import datetime
from ads.opctl.operator.cmd import run


DATASETS_LIST = [
    "AirPassengersDataset",
    "AusBeerDataset",
    "AustralianTourismDataset",
    "ETTh1Dataset",
    # 'ETTh2Dataset',
    # 'ETTm1Dataset',
    # 'ETTm2Dataset',
    # 'ElectricityDataset',
    "EnergyDataset",
    "ExchangeRateDataset",
    "GasRateCO2Dataset",
    "HeartRateDataset",
    "ILINetDataset",
    "IceCreamHeaterDataset",
    "MonthlyMilkDataset",
    "MonthlyMilkIncompleteDataset",
    "SunspotsDataset",
    "TaylorDataset",
    "TemperatureDataset",
    "TrafficDataset",
    "USGasolineDataset",
    "UberTLCDataset",
    "WeatherDataset",
    "WineDataset",
    "WoolyDataset",
]

MODELS = [
    "arima",
    "automlx",
    "prophet",
    "neuralprophet",
    "autots",
    # "auto",
]

TEMPLATE_YAML = {
    "kind": "operator",
    "type": "forecast",
    "version": "v1",
    "spec": {
        "historical_data": {
            "url": None,
        },
        "output_directory": {
            "url": "results",
        },
        "model": None,
        "target_column": None,
        "datetime_column": {
            "name": None,
        },
        "target_category_columns": [],
        "horizon": None,
        "generate_explanations": False,
    },
}


PERIODS = 5
MAX_ADDITIONAL_COLS = 3
SAMPLE_FRACTION = 1

parameters_short = []

for dataset_i in DATASETS_LIST[1:3]:  #  + [DATASETS_LIST[-2]]
    for model in MODELS:
        parameters_short.append((model, dataset_i))


@pytest.mark.parametrize("model, dataset_name", parameters_short)
def test_load_datasets(model, dataset_name):
    dataset_i = getattr(d_datasets, dataset_name)().load()
    datetime_col = dataset_i.time_index.name

    columns = dataset_i.components
    target = dataset_i[columns[0]][:-PERIODS]
    test = dataset_i[columns[0]][-PERIODS:]

    print(dataset_name, len(columns), len(target))
    with tempfile.TemporaryDirectory() as tmpdirname:
        historical_data_path = f"{tmpdirname}/primary_data.csv"
        additional_data_path = f"{tmpdirname}/add_data.csv"
        test_data_path = f"{tmpdirname}/test_data.csv"
        output_data_path = f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        generate_train_metrics = True  # bool(random.getrandbits(1))

        df_i = target.pd_dataframe().reset_index()
        if dataset_name == "AustralianTourismDataset":
            df_i[datetime_col] = [
                pd.to_datetime("1990-01-01", format="%Y-%m-%d")
            ] + df_i[datetime_col].values * datetime.timedelta(days=1)

        df_i.to_csv(historical_data_path, index=False)
        # .sample(frac=SAMPLE_FRACTION).sort_values(by=datetime_col)

        test_df = test.pd_dataframe().reset_index()
        if dataset_name == "AustralianTourismDataset":
            test_df[datetime_col] = [
                pd.to_datetime("1990-01-01", format="%Y-%m-%d")
            ] + test_df[datetime_col].values * datetime.timedelta(days=1)

        test_df.to_csv(test_data_path, index=False)

        if len(columns) > 1:
            additional_cols = columns[1 : min(len(columns), MAX_ADDITIONAL_COLS)]
            additional_data = dataset_i[list(additional_cols)]
            df_additional = additional_data.pd_dataframe().reset_index()
            if dataset_name == "AustralianTourismDataset":
                df_additional[datetime_col] = [
                    pd.to_datetime("1990-01-01", format="%Y-%m-%d")
                ] + df_additional[datetime_col].values * datetime.timedelta(days=1)

            df_additional.to_csv(additional_data_path, index=False)
            yaml_i["spec"]["additional_data"] = {"url": additional_data_path}

        yaml_i["spec"]["historical_data"]["url"] = historical_data_path
        yaml_i["spec"]["test_data"] = {"url": test_data_path}
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["target_column"] = columns[0]
        yaml_i["spec"]["datetime_column"]["name"] = datetime_col
        yaml_i["spec"]["horizon"] = PERIODS
        if yaml_i["spec"].get("additional_data") is not None:
            yaml_i["spec"]["generate_explanations"] = True
        if generate_train_metrics:
            yaml_i["spec"]["generate_metrics"] = generate_train_metrics
        if model == "autots":
            yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 1}

        forecast_yaml_filename = f"{tmpdirname}/forecast.yaml"
        # with open(f"{tmpdirname}/forecast.yaml", "w") as f:
        #     f.write(yaml.dump(yaml_i))
        # sleep(0.5)
        # subprocess.run(
        #     f"ads operator run -f {forecast_yaml_filename} --debug", shell=True  #  --debug
        # )
        # sleep(0.1)
        run(yaml_i, backend="local", debug=True)
        subprocess.run(f"ls -a {output_data_path}", shell=True)
        if yaml_i["spec"]["generate_explanations"]:
            glb_expl = pd.read_csv(f"{tmpdirname}/results/global_explanation.csv")
            print(glb_expl)
            loc_expl = pd.read_csv(f"{tmpdirname}/results/local_explanation.csv")
            print(loc_expl)

        test_metrics = pd.read_csv(f"{tmpdirname}/results/test_metrics.csv")
        print(test_metrics)
        train_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
        print(train_metrics)
        return test_metrics.iloc[0][f"{columns[0]}"]


def run_operator(
    historical_data_path,
    additional_data_path,
    test_data_path,
    generate_train_metrics=True,
    output_data_path=None,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_data_path = output_data_path or f"{tmpdirname}/results"
        yaml_i = deepcopy(TEMPLATE_YAML)
        generate_train_metrics = True

        yaml_i["spec"]["additional_data"] = {"url": additional_data_path}
        yaml_i["spec"]["historical_data"]["url"] = historical_data_path
        yaml_i["spec"]["test_data"] = {"url": test_data_path}
        yaml_i["spec"]["output_directory"]["url"] = output_data_path
        yaml_i["spec"]["model"] = model
        yaml_i["spec"]["target_column"] = "Sales"
        yaml_i["spec"]["datetime_column"]["name"] = "Date"
        yaml_i["spec"]["target_category_columns"] = ["Store"]
        yaml_i["spec"]["horizon"] = PERIODS

        if generate_train_metrics:
            yaml_i["spec"]["generate_metrics"] = generate_train_metrics
        if model == "autots":
            yaml_i["spec"]["model_kwargs"] = {"model_list": "superfast"}
        if model == "automlx":
            yaml_i["spec"]["model_kwargs"] = {"time_budget": 1}

        forecast_yaml_filename = f"{tmpdirname}/forecast.yaml"
        with open(f"{tmpdirname}/forecast.yaml", "w") as f:
            f.write(yaml.dump(yaml_i))
        sleep(0.1)
        subprocess.run(
            f"ads operator run -f {forecast_yaml_filename} --debug", shell=True
        )
        sleep(0.1)
        subprocess.run(f"ls -a {output_data_path}", shell=True)

        test_metrics = pd.read_csv(f"{tmpdirname}/results/test_metrics.csv")
        print(test_metrics)
        train_metrics = pd.read_csv(f"{tmpdirname}/results/metrics.csv")
        print(train_metrics)


# parameters_datetime = []
# DATETIME_FORMATS_TO_TEST = [
#     ["%Y", datetime.timedelta()],
#     ["%y", ],
#     ["%b-%d-%Y",],
#     ["%d-%m-%y",],
#     ["%d/%m/%y %H:%M:%S",],
# ]

# for dt_format in DATETIME_FORMATS_TO_TEST:
#     for (model) in MODELS:
#         parameters_datetime.append((model, dt_format))


# @pytest.mark.parametrize("model, dt_format", parameters_datetime)
# def test_datetime_formats(model=model, dt_format=dt_format):
#     curr_dir = pathlib.Path(__file__).parent.resolve()
#     data_folder = f"{curr_dir}/../data/"
#     np.arrange((1000, 12))
#     d1 = np.random.multivariate_normal(
#         mean=np.array([-0.5, 0, 2]),
#         cov=np.array([[1, 0, 0.5], [0, 1, 0.7], [0.5, 0.7, 1]]),
#         size=len,
#     )
#     now = datetime.datetime.now()
#     now_formatted = now.strftime(dt_format)

#     historical_data_path = f"{data_folder}/rs_10_prim.csv"
#     additional_data_path = f"{data_folder}/rs_10_add.csv"
#     test_data_path = f"{data_folder}/rs_10_test.csv"

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         output_data_path = f"{tmpdirname}/results"
#         yaml_i = deepcopy(TEMPLATE_YAML)
#         generate_train_metrics = True


if __name__ == "__main__":
    failed_runs = []
    results = dict()
    timings = dict()
    for dataset_name in DATASETS_LIST[2:3]:  # random.sample(DATASETS_LIST, 2):
        results[dataset_name] = dict()
        timings[dataset_name] = dict()
        for m in [
            "automlx"
        ]:  # ["arima", "automlx", "prophet", "neuralprophet", "autots", "auto"]:
            start_time = time()
            try:
                results[dataset_name][m] = test_load_datasets(
                    model=m, dataset_name=dataset_name
                )
            except Exception as e:
                print(f"Failed with the following error! {e}")
                failed_runs.append((dataset_name, m))
            elapsed = time() - start_time
            timings[dataset_name][m] = elapsed
    print(f"Failed Runs: {failed_runs}")
    print(f"results: {pd.DataFrame(results)}")
    print(f"timings: {timings}")
    pd.DataFrame(results).to_csv("~/Desktop/AUTO_benchmark_darts.csv")
