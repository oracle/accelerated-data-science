#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
import datapane as dp
from prophet.plot import add_changepoints_to_plot
from prophet import Prophet
from neuralprophet import NeuralProphet as Prophet
from neuralprophet.plot import add_changepoints_to_plot
import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ForecastOperator:
    def __init__(self, **kwargs):
        self.input_filename = "pypistats.csv"
        self.report_filename = "report.html"
        self.output_filename = "output.csv"
        self.ds_column = "date"
        self.datetime_format = None
        self.target_columns = ["ocifs_downloads", "oracle-ads_downloads", "oci-mlflow_downloads"]

        self.horizon = {
            "periods": 31,
            "interval": 1,
            "interval_unit": "D",
        }

    def load_data(self):
        # Load data and format datetime column
        data = pd.read_csv(self.input_filename)
        data["ds"] = pd.to_datetime(data[self.ds_column], format=self.datetime_format)
        data.drop([self.ds_column], axis=1, inplace=True)
        data.fillna(0, inplace=True)
        self.data = data

        models = []
        outputs = []
        for i, col in enumerate(self.target_columns):
            data_i = data[[col, "ds"]]
            print(f"using columns: {data_i.columns}")
            data_i.rename({col:"y"}, axis=1, inplace=True)
            
            model = Prophet()
            # Add regressors
            # Add metrics
            # Use forecasting service datasets
            # report should have html colored code for yaml file
            model.fit(data_i)

            future = model.make_future_dataframe(periods=self.horizon['periods']) #, freq=self.horizon['interval_unit']
            forecast = model.predict(future)

            print(f"-------Model {i}----------------------")
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            models.append(model)
            outputs.append(forecast)
        
        self.models = models
        self.outputs = outputs

        print("===========Done===========")
        output_total = pd.concat(self.outputs).to_csv(self.output_filename)
        return self.outputs

    def generate_report(self):
        def get_select_plot_list(fn):
            return dp.Select(blocks=[dp.Plot(fn(i), label=col) for i, col in enumerate(self.target_columns)])

        title_text = dp.Text("# Forecast Report")
        sec1_text = dp.Text(f"## Forecast Overview")
        sec1 = get_select_plot_list(lambda idx: self.models[idx].plot(self.outputs[idx], include_legend=True))
        sec2_text = dp.Text(f"## Forecast Broken Down by Trend Component")
        sec2 = get_select_plot_list(lambda idx: self.models[idx].plot_components(self.outputs[idx]))

        sec3_text = dp.Text(f"## Forecast Changepoints")
        sec3_figs = [self.models[idx].plot(self.outputs[idx]) for idx in range(len(self.target_columns))]
        [add_changepoints_to_plot(sec3_figs[idx].gca(), self.models[idx], self.outputs[idx]) for idx in range(len(self.target_columns))]
        sec3 = get_select_plot_list(lambda idx: sec3_figs[idx])
        sec4_text = dp.Text(f"## Forecast Seasonality Parameters")
        sec4 = dp.Select(blocks=[dp.Table(pd.DataFrame(m.seasonalities), label=self.target_columns[i]) for i, m in enumerate(self.models)])

        self.view = dp.View(title_text, sec1_text, sec1, sec2_text, sec2, sec3_text, sec3, sec4_text, sec4)
        dp.save_report(self.view, self.report_filename, open=True)
        print(f"Generated Report: {self.report_filename}")
        return self.view


def operate(args):
    operator = ForecastOperator(**args)
    forecasts = operator.load_data()
    report = operator.generate_report()
    return forecasts

    # Return fully verbose yaml
    # Offer some explanations
    # Reccomend other possible models

if __name__ == '__main__':
    operate(dict())



    # from urllib.parse import urlparse

    # import ads
    # import numpy
    # import pandas as pd
    # from sklearn.metrics import classification_report
    # from sklearn.model_selection import train_test_split

    # import oci
    # import time
    # from datetime import datetime

#     time_budget = args.get("max_runtime", 100)
#     keep_features = args.get("keep_features", 1)
#     task = args.get("task")
#     test_size = args.get("test_size", 0.1)
#     exclude_features = args.get("exclude_features", [])
#     target_feature = args.get("target")

#     print(f"{target_feature=}, {exclude_features=}, {keep_features=}")
#     print(f"{test_size=}")

#     source = args.get("source")

#     auth_type = os.environ.get("OCIFS_IAM_TYPE", "resource_principal")
#     print(f"Setting auth type as {auth_type}")
#     ads.set_auth(auth_type)

#     print(f"Loading data from {source['url']}")
#     df = pd.read_csv(source["url"])

#     categorical_features = df.select_dtypes(exclude=["int", "float"]).columns
#     for feature in categorical_features:
#         df[feature] = df[feature].astype("category")
#     print(f"Column dtypes are: {df.dtypes}")

#     X_train, X_test, y_train, y_test = train_test_split(
#         df[df.columns.difference([target_feature] + exclude_features)],
#         df[target_feature],
#         test_size=test_size,
#     )

#     init(engine="local")

#     print(f"Setting up automl for training")
#     est = automl.Pipeline(
#         task=task, model_list=["LGBMClassifier"], min_features=keep_features
#     )

#     print(f"Starting training")
#     est.fit(X_train, y_train)

#     print(f"Evaluating the model")
#     y_pred = est.predict(X_test)

#     print(classification_report(y_test, y_pred))

#     print(f"Generating explanation and evaluation report...")

#     exp_artifact_dir = "model_artifact_exp"
#     artifact_dir = "model_artifact"
#     report_name = args.get("report_file_name", "report.html")

#     os.makedirs(artifact_dir)

#     params = {
#         key: int(est.selected_model_params_[key])
#         if isinstance(est.selected_model_params_[key], numpy.int64)
#         else est.selected_model_params_[key]
#         for key in est.selected_model_params_
#     }
#     classifier = lgbm.LGBMClassifier(**params)
#     classifier.fit(
#         X_train,
#         y_train,
#         categorical_feature="auto",
#     )
#     protected_features = args.get("protected_features", None)

#     lgbm_model = LightGBMModel(
#         classifier, artifact_dir=exp_artifact_dir, force_overwrite=True
#     )
#     output = lgbm_model.prepare(
#         inference_conda_env="generalml_p38_cpu_v1",
#         training_conda_env="generalml_p38_cpu_v1",
#         X_sample=X_train,
#         y_sample=y_train,
#         use_case_type=UseCaseType.BINARY_CLASSIFICATION,
#         force_overwrite=True,
#     )

#     print("Generating model 360 report..")
#     print("protected features: {protected_features}")
#     model360report = lgbm_model.report(
#         positive_class=1,
#         protected_features=protected_features[0] if protected_features else None,
#     )

#     report_path = os.path.join(artifact_dir, report_name)
#     model360report.save(report_path)

#     # Saving model
#     from ads.model import GenericModel

#     model = GenericModel(est, artifact_dir=artifact_dir, force_overwrite=True)

#     conda_bucket = os.environ.get("CONDA_ENV_BUCKET")
#     conda_namespace = os.environ.get("CONDA_ENV_NAMESPACE")
#     conda_path = os.environ.get("CONDA_ENV_OBJECT_NAME")

#     model.prepare(
#         inference_conda_env=f"oci://{conda_bucket}@{conda_namespace}/{conda_path}",
#         training_conda_env=f"oci://{conda_bucket}@{conda_namespace}/{conda_path}",
#         X_sample=X_train,
#         y_sample=y_train,
#         use_case_type=UseCaseType.BINARY_CLASSIFICATION,
#         inference_python_version="3.8",
#         force_overwrite=True,
#     )

#     with open(os.path.join(artifact_dir, "score.py"), "w") as sf:
#         with open(os.path.join(os.path.dirname(__file__), "score.py")) as of:
#             sf.write(of.read())
#     print("Verify artifacts..")
#     print(model.verify(X_test[:3].to_json()))
#     print("Successfully Verified")

#     os_client = ads.common.oci_client.OCIClientFactory(
#         **ads.auth.default_signer()
#     ).object_storage
#     output_dir = args.get("output_location")

#     print(f"Uploading report to {output_dir}")

#     parsed_path = urlparse(output_dir)
#     report_object_name = os.path.join(parsed_path.path[1:], report_name)
#     response = os_client.put_object(
#         namespace_name=parsed_path.hostname,
#         bucket_name=parsed_path.username,
#         object_name=report_object_name,
#         put_object_body=open(report_path, "rb").read(),
#         content_disposition="text/html",
#     )

#     response = os_client.create_preauthenticated_request(
#         namespace_name=parsed_path.hostname,
#         bucket_name=parsed_path.username,
#         create_preauthenticated_request_details=oci.object_storage.models.CreatePreauthenticatedRequestDetails(
#             name=f"oracle_automl_report_{int(time.time())}",
#             access_type="ObjectReadWrite",
#             time_expires=datetime.strptime(
#                 "2043-03-01T02:20:55.385Z", "%Y-%m-%dT%H:%M:%S.%fZ"
#             ),
#             bucket_listing_action="Deny",
#             object_name=report_object_name,
#         ),
#     )

#     upload_location = f"{os_client.base_client.endpoint}{response.data.access_uri}"

#     print(f"report uploaded to {upload_location}")
#     model.metadata_custom.add(
#         key="model_report",
#         value=upload_location,
#         category=MetadataCustomCategory.OTHER,
#         description="Comprehensive model report",
#         replace=True,
#     )

#     with ads.model.experiment(
#         name=args.get(
#             "model_version_set_name", f"AutoML Model: {args.get('model_name')}"
#         ),
#         create_if_not_exists=True,
#     ):

#         display_name = f"AutoML Model: {args.get('model_name')}"
#         model.save(
#             display_name=display_name,
#         )
#         print(model.dsc_model)

#     print(f"Model saved succesfully.")


# def run():
#     args = json.loads(os.environ.get("OPERATOR_ARGS", "{}"))
#     operate(args)
