#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import time
import numpy as np

import pytest
from ads.model.deployment.common.utils import State as ModelDeploymentState
from ads.model.deployment.model_deployer import ModelDeployer
from ads.model.generic_model import (
    GenericModel,
    VERIFY_STATUS_NAME,
    PREPARE_STATUS_NAME,
)
from ads.model.framework.tensorflow_model import TensorFlowModel
from ads.model.framework.pytorch_model import PyTorchModel
from ads.model.framework.xgboost_model import XGBoostModel
from ads.model.framework.sklearn_model import SklearnModel
from ads.model.framework.lightgbm_model import LightGBMModel
from ads.model.framework.spark_model import SparkPipelineModel
from ads.evaluations.evaluator import Evaluator
from scripts.time_series_model import timeseries
from scripts.pytorch import pytorch
from scripts.tensorflow import tensorflow
from scripts.spark_pipeline import (
    spark_pipeline_script1,
    spark_pipeline_script2,
    spark_pipeline_script3,
)
from scripts.sklearn import (
    sklearn_no_pipeline,
    sklearn_pipeline_with_sklearn_model,
    sklearn_pipeline_with_xgboost_model,
    sklearn_pipeline_with_lightgbm_model,
)
from scripts.lightgbm_and_xgboost import (
    lightgbm_learning_api,
    lightgbm_sklearn_api_regression,
    lightgbm_sklearn_api_classification,
    xgboost_learning_api,
    xgboost_sklearn_api_regression,
    xgboost_sklearn_api_classification,
)
from oci.exceptions import ServiceError
from tests.integration.config import secrets

DIFF = 0.0001

MODEL_PARAMS = [
    sklearn_no_pipeline(),
    sklearn_pipeline_with_sklearn_model(),
    sklearn_pipeline_with_xgboost_model(),
    sklearn_pipeline_with_lightgbm_model(),
    spark_pipeline_script1(),
    spark_pipeline_script2(),
    spark_pipeline_script3(),
    lightgbm_learning_api(),
    lightgbm_sklearn_api_regression(),
    lightgbm_sklearn_api_classification(),
    xgboost_sklearn_api_regression(),
    xgboost_learning_api(),
]


class TestFramework:
    model_params = MODEL_PARAMS + [
        xgboost_sklearn_api_classification(),
        timeseries(),
        pytorch(),
        tensorflow(),
    ]
    models = []
    deployer = ModelDeployer()

    def setup_class(cls):
        network_compartment_id = secrets.common.COMPARTMENT_ID
        TC_project_id = secrets.common.PROJECT_OCID

        TEST_compartment_id = secrets.other.COMPARTMENT_ID
        TEST2_project_id = secrets.other.PROJECT_ID
        num_deployments = 0
        for status in [ModelDeploymentState.ACTIVE, ModelDeploymentState.FAILED]:
            for compartment_id, project_id in zip(
                [network_compartment_id, TEST_compartment_id],
                [TC_project_id, TEST2_project_id],
            ):
                deployments = cls.deployer.list_deployments(
                    compartment_id=compartment_id, project_id=project_id, status=status
                )
                num_deployments += len(deployments)
                for deployment in deployments:
                    cls.deployer.delete(
                        model_deployment_id=deployment.model_deployment_id,
                        wait_for_completion=False,
                    )

        if num_deployments > 0 and num_deployments <= 5:
            time.sleep(3 * 60)
        elif num_deployments > 5:
            time.sleep(6 * 60)

    @pytest.mark.parametrize("params", MODEL_PARAMS)
    def test_class_names(self, params):
        framework = params["framework"]
        if framework in [PyTorchModel, TensorFlowModel, GenericModel]:
            return
        estimator = params["estimator"]
        artifact_dir = params["artifact_dir"]
        X = params["data"]
        y = params["y_true"]
        y_pred = params["local_pred"]
        model = framework(estimator=estimator, artifact_dir=artifact_dir)
        with pytest.raises(ValueError):
            evaluator = Evaluator(models=[model], X=X, y=y)
            raw_html = evaluator.html()

        df = model.summary_status().reset_index()
        assert (
            df.loc[df["Step"] == PREPARE_STATUS_NAME, "Status"] == "Available"
        ).all()
        if framework not in [SparkPipelineModel]:
            model.prepare(
                inference_conda_env=params["inference_conda_env"],
                inference_python_version=params["inference_python_version"],
                model_file_name=params["model_file_name"],
                force_overwrite=True,
                prepare_args=(
                    params["prepare_args"] if "prepare_args" in params else {}
                ),
            )
            df = model.summary_status().reset_index()
            assert (df.loc[df["Step"] == PREPARE_STATUS_NAME, "Status"] == "Done").all()

            assert (
                df.loc[df["Step"] == VERIFY_STATUS_NAME, "Status"] == "Available"
            ).all()
            with pytest.raises(ValueError):
                evaluator = Evaluator(models=[model], X=X, y=y)

        model.prepare(
            inference_conda_env=params["inference_conda_env"],
            inference_python_version=params["inference_python_version"],
            model_file_name=params["model_file_name"],
            force_overwrite=True,
            X_sample=X,
            y_sample=y,
            prepare_args=(params["prepare_args"] if "prepare_args" in params else {}),
        )
        evaluator = Evaluator(models=[model], X=X, y=y)
        raw_html = evaluator.html()
        assert raw_html is not None
        evaluator.display()
        evaluator.save("report.html")

    def test_pipeline(self):
        i = -1
        j = 0
        for _, params in enumerate(self.model_params):
            for as_onnx in [True, False]:
                framework = params["framework"]
                estimator = params["estimator"]
                artifact_dir = params["artifact_dir"] + "_as_onnx_" + str(as_onnx)
                print(f"starting framework {framework}")
                print(f"as_onnx is {as_onnx}")
                print(f"esimtator is {estimator.__class__.__name__}")
                print(artifact_dir)

                if framework in [GenericModel, SparkPipelineModel] and as_onnx:
                    continue
                i += 1

                inference_conda_env = params["inference_conda_env"]
                inference_python_version = params["inference_python_version"]
                model_file_name = params["model_file_name"]
                data = params["onnx_data"] if as_onnx else params["data"]
                local_pred = params["local_pred"]
                score_py_path = params["score_py_path"]
                prepare_args = (
                    params["prepare_args"] if "prepare_args" in params else {}
                )

                model = framework(estimator=estimator, artifact_dir=artifact_dir)
                assert model.artifact_dir == os.path.abspath(
                    os.path.expanduser(artifact_dir)
                )
                df = model.summary_status().reset_index()
                assert (df.loc[df["Step"] == "initiate", "Status"] == "Done").all()
                assert (df.loc[df["Step"] == "initiate", "Actions Needed"] == "").all()

                assert (df.loc[df["Step"] == "prepare", "Status"] == "Available").all()

                model.prepare(
                    inference_conda_env=inference_conda_env,
                    model_file_name=model_file_name,
                    inference_python_version=inference_python_version,
                    force_overwrite=True,
                    training_id=None,
                    X_sample=params["data"] if as_onnx else None,
                    as_onnx=as_onnx,
                    **prepare_args,
                )

                df = model.summary_status().reset_index()
                assert (df.loc[df["Step"] == "prepare", "Status"] == "Done").all()

                assert (df.loc[df["Step"] == "save", "Status"] == "Available").all()
                assert (df.loc[df["Step"] == "verify", "Status"] == "Available").all()

                assert (
                    df.loc[df["Step"] == "deploy", "Status"] == "Not available"
                ).all()
                assert (
                    df.loc[df["Step"] == "predict", "Status"] == "Not available"
                ).all()

                if score_py_path and not as_onnx:
                    with open(
                        os.path.join(os.path.dirname(__file__), score_py_path)
                    ) as fr:
                        with open(
                            os.path.join(model.artifact_dir, "score.py"), mode="w"
                        ) as fw:
                            fw.write(fr.read())

                verify_pred = model.verify(data)
                df = model.summary_status().reset_index()
                assert (df.loc[df["Step"] == "verify", "Status"] == "Done").all()
                # Testing .verify()
                self.compare_with_local_pred(
                    verify_pred, local_pred, framework, as_onnx, estimator
                )

                model.save(timeout=3000)
                model.summary_status()
                try:
                    model.deploy(
                        wait_for_completion=False,
                        display_name=artifact_dir.split("/")[-1],
                    )
                except ServiceError as e:
                    print("Service Limit exceeded. Delete some deployments first.")
                    time.sleep(5 * 60)
                    model.deploy(wait_for_completion=False)
                self.models.append(
                    (
                        artifact_dir,
                        model,
                        data,
                        local_pred,
                        framework,
                        as_onnx,
                        estimator,
                    )
                )
                n = 6
                if (i + 1 > 0 and (i + 1) % n == 0) or (i + 1) == (
                    len(self.model_params) * 2 - 1
                ):
                    # Time to wait the deployment become active.
                    time.sleep(30 * 60)

                    for m in range(j, i + 1):
                        (
                            name,
                            deployed_model,
                            test_data,
                            local_model_pred,
                            framework,
                            as_onnx,
                            estimator,
                        ) = self.models[m]
                        print("deployment predict starts for: ", name)
                        try:
                            deployment_pred = deployed_model.predict(test_data)
                            self.compare_with_local_pred(
                                deployment_pred,
                                local_model_pred,
                                framework,
                                as_onnx,
                                estimator,
                            )
                        except Exception as e:
                            raise e
                        finally:
                            try:
                                print("Deleting tested model deployment:", name)
                                deployed_model.delete_deployment()
                            except:
                                pass
                    try:
                        self.clean_md()
                    except:
                        pass
                    j = i + 1
                    # Time to wait the tested model deployment has been deleted.
                    time.sleep(5 * 60)

    def clean_md(self):
        for status in [
            ModelDeploymentState.ACTIVE,
            ModelDeploymentState.FAILED,
        ]:
            deployments = self.deployer.list_deployments(
                compartment_id=self.network_compartment_id,
                project_id=self.TC_project_id,
                status=status,
            )

            for deployment in deployments:
                self.deployer.delete(
                    model_deployment_id=deployment.model_deployment_id,
                    wait_for_completion=False,
                )

    def compare_with_local_pred(
        self, test_pred, local_pred, framework, as_onnx, estimator
    ):
        if framework == PyTorchModel:
            self.pytorch_pred_compare(test_pred, local_pred)
        elif framework == TensorFlowModel:
            self.tensorflow_pred_compare(test_pred, local_pred)
        elif (
            framework == XGBoostModel
            and not as_onnx
            and "Classifier" in estimator.__class__.__name__
        ):
            assert np.array(local_pred) == pytest.approx(
                np.argmax(np.array(test_pred["prediction"]), axis=1)
            )
        elif (
            (
                framework == XGBoostModel
                or framework == LightGBMModel
                or framework == SklearnModel
            )
            and as_onnx
            and (
                "Classifier" in estimator.__class__.__name__
                or "LogisticRegression" in estimator.__class__.__name__
            )
        ):
            assert np.array(local_pred) == pytest.approx(
                np.array(test_pred["prediction"][0]).reshape(local_pred.shape)
            )
        elif str(estimator.__class__) == "<class 'lightgbm.basic.Booster'>" and as_onnx:
            local = [0 if x <= 0.5 else 1 for x in local_pred]
            assert (np.array(test_pred["prediction"]) - np.array(local) == 0).all()
        else:
            assert np.array(local_pred) == pytest.approx(
                np.array(test_pred["prediction"]).reshape(local_pred.shape),
                abs=0.1,
            )

    def tensorflow_pred_compare(self, pred1, pred2):
        assert isinstance(pred1, dict)
        assert len(pred1["prediction"][0]) == pred2.shape[1]
        for i in range(pred2.shape[1]):
            assert abs(pred1["prediction"][0][i] - pred2[0][i]) < DIFF

    def pytorch_pred_compare(self, pred1, pred2):
        assert isinstance(pred1, dict)
        assert len(pred1["prediction"][0]) == pred2.shape[1]
        assert len(pred1["prediction"][0][0]) == pred2.shape[2]
        assert len(pred1["prediction"][0][0][0]) == pred2.shape[3]

        for i in range(10):
            assert abs(pred1["prediction"][0][0][0][i] - pred2[0][0][0][i]) < DIFF

    def teardown_class(self):
        shutil.rmtree("./artifact_folder", ignore_errors=True)
