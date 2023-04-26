#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import json
import os
import pickle
import shutil
import tempfile
import uuid

from ads.common import oci_client
from ads.common import auth as authutil

import sklearn
from ads.hpo.ads_search_space import model_list
from ads.hpo.distributions import decode, encode
from ads.hpo.utils import _extract_uri

from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class NotPickableError(Exception):   # pragma: no cover
    def __init__(self, message):
        self.message = message


class UploadTunerArtifact:
    def __init__(self, tuner, file_uri, metadata, auth=None):
        self.tuner = tuner
        self.file_uri = file_uri
        self.metadata = metadata
        self.oci_client = None
        self.tuner_args = None
        self.zip_dir = None
        self.auth = auth if auth else authutil.default_signer()

    def upload(self, script_dict):
        tuner_zip = self.prepare_tuner_artifact(script_dict)
        self.upload_to_cloud(tuner_zip)

    def prepare_tuner_artifact(self, script_dict):
        """
        zip and save all the tuner files

        Args:
            script_dict (dict): dict which contains the script names and path
        """
        tuner_path = "tuner_artifacts" + str(uuid.uuid4()) + ".zip"
        tuner_path_name = os.path.splitext(tuner_path)[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            # make the zip dir is not exist
            self.zip_dir = os.path.join(temp_dir, tuner_path_name)
            logger.debug(f"create the temp dir: {self.zip_dir}")
            if not os.path.exists(self.zip_dir):
                os.makedirs(self.zip_dir)
            logger.debug(
                f"start serializing the tuner argument and save at tuner.json ..."
            )
            self.tuner_args = self.json_serialize_tuner_args(script_dict)
            logger.debug(f"start saving tuner.json at {tuner_path}...")
            self._save_tuner_args_as_json()
            logger.debug(f"start saving the scipts at {tuner_path} if there are any...")
            self._save_script(script_dict)
            if "sqlite:///" in self.tuner.storage:
                if "sqlite:///:memory:" in self.tuner.storage:
                    raise NotImplementedError(
                        "Cannot save in-memory sqlite database to the object storage. Store it on block storage instead."
                    )
                shutil.copy(
                    os.path.join(self.tuner.storage.replace("sqlite:///", "")),
                    self.zip_dir,
                )
            else:
                raise NotImplementedError(
                    "Only sqlite file is supported to be saved to the object storage."
                )
            logger.debug(f"start archiving the files at {tuner_path}...")
            shutil.make_archive(
                tuner_path_name, "zip", base_dir=tuner_path_name, root_dir=temp_dir
            )
        return tuner_path

    def json_serialize_tuner_args(self, script_dict):
        """
        json serialize the tuner args
        """
        tuner_args = {}
        serialising_fields = [
            "model",
            "scoring",
            "strategy",
            "cv",
            "loglevel",
            "storage",
            "study_name",
            "__random_state",
            "load_if_exists",
            "n_jobs",
        ]
        for field in serialising_fields:
            if field == "strategy":
                tuner_args["strategy"] = encode(self.tuner.strategy)
            elif field == "model":
                tuner_args["model"] = self._serialize_model(script_dict)
            elif field == "scoring" and self.tuner.scoring is not None:
                tuner_args["scoring"] = self._serialize_scoring(script_dict)
            else:
                try:
                    if getattr(self.tuner, field) is not None:
                        tuner_args[field] = getattr(self.tuner, field)
                except:
                    logger.debug(
                        field.replace("__", "") + " is not serialized since it is None."
                    )
        return tuner_args

    def _serialize_model(self, script_dict):
        if "model" in script_dict and script_dict["model"] is not None:
            # save as script
            model = {"model_script": script_dict["model"]}
        elif self.tuner.model.__class__ in model_list:
            model = {
                "model_name": self.tuner.model.__class__.__name__,
                "model_input": self.tuner.model.get_params(),
            }
        else:
            # save as pickle
            try:
                model = {"pickled_model_path": "model.pkl"}
                with open(
                    os.path.join(self.zip_dir, model["pickled_model_path"]), "wb"
                ) as f:
                    pickle.dump(self.tuner.model, f)
            except:
                raise NotPickableError(
                    "Model is not pickable. Try to save the model in a script."
                )
        return model

    def _serialize_scoring(self, script_dict):
        if isinstance(self.tuner.scoring, str):
            scoring = {"scoring_name": self.tuner.scoring}
        else:
            assert (
                script_dict["scoring"] is not None
            ), "Need to save the customized scoring function \
                    in a script and pass the script name to <code/>script_dict={'scoring': script_name}<code>"
            scoring = {"scoring_script": script_dict["scoring"]}
        return scoring

    def _save_script(self, script_dict):
        for var, script_path in script_dict.items():
            if script_path is not None:
                # script_name = os.path.basename(script_path)
                #
                shutil.copy(script_path, self.zip_dir)

    def _save_tuner_args_as_json(self):
        try:
            with open(os.path.join(self.zip_dir, "tuner.json"), "w") as outfile:
                json.dump(self.tuner_args, outfile)
        except:
            raise RuntimeError(f"Failed to save .json at this path: {self.zip_dir}.")

    def upload_to_cloud(self, tuner_zip):
        bucketname, namespace, filename = _extract_uri(self.file_uri)
        if self.oci_client is None:
            self.oci_client = oci_client.OCIClientFactory(**self.auth).object_storage

        try:
            with open(tuner_zip, "rb") as zipfile:
                self.oci_client.put_object(
                    namespace,
                    bucketname,
                    filename,
                    zipfile,
                    opc_meta={"info": self.metadata},
                )
        finally:
            logger.debug(f"removing {tuner_zip}")
            os.remove(tuner_zip)


class DownloadTunerArtifact:
    """
    Download the tuner artifact from the cloud and deserialize the tuner args
    """

    def __init__(self, file_uri, target_file_path=None, auth=None):
        self.file_uri = file_uri
        self.target_file_path = (
            os.path.join("/tmp", "hpo_" + str(uuid.uuid4()) + ".zip")
            if target_file_path is None
            else target_file_path
        )
        self.oci_client = None
        self.tuner_args = None
        self.metadata = None
        self.auth = auth if auth else authutil.default_signer()

    def extract_tuner_args(self, delete_zip_file=False):
        """
        deserialize tuner argument from the zip file
        """
        self.download_from_cloud()
        file_path_dict = self._find_file_path()
        tuner_args = self._load_json(file_path_dict["json_path"])
        if tuner_args is not None:
            self._save_db_file(file_path_dict)
            transformed_args = self.deserialize_tuner_args(file_path_dict)
            if delete_zip_file:
                os.remove(self.target_file_path)
                shutil.rmtree(os.path.splitext(self.target_file_path)[0])
            return transformed_args, self.metadata
        else:
            raise RuntimeError("Tuner args are not found or loaded.")

    def download_from_cloud(self):
        """
        Download the artifact and unpack the arhchive at the target file path
        """
        if os.path.isdir(self.target_file_path):
            raise ValueError(
                "`target_file_path` should be a file, for example, '/home/datascience/myfile.zip', but given a dir."
            )
        if not self.target_file_path.endswith(".zip"):
            self.target_file_path += ".zip"
        bucketname, namespace, filename = _extract_uri(self.file_uri)
        if self.oci_client is None:
            self.oci_client = oci_client.OCIClientFactory(**self.auth).object_storage

        res = self.oci_client.get_object(namespace, bucketname, filename)
        self.metadata = res.data.headers["opc-meta-info"]

        if os.path.exists(self.target_file_path):
            raise FileExistsError(
                "This file name already exists, do you want to delete it or use a different file path?"
            )
        target_dir = os.path.dirname(self.target_file_path)
        if not os.path.exists(target_dir):
            os.path.makedirs(target_dir)
        with open(self.target_file_path, "wb") as dbfile:
            for chunk in res.data.iter_content(chunk_size=4096):
                dbfile.write(chunk)
        shutil.unpack_archive(
            self.target_file_path, os.path.splitext(self.target_file_path)[0]
        )

    def _find_file_path(self):
        file_path_dict = {}
        for subdir, dirs, files in os.walk(os.path.splitext(self.target_file_path)[0]):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file.endswith(".json"):
                    file_path_dict["json_path"] = file_path
                if file.endswith(".db"):
                    file_path_dict["db_path"] = file_path
        return file_path_dict

    def _load_json(self, json_path):
        if os.path.exists(json_path):
            with open(json_path, "r") as outfile:
                self.tuner_args = json.load(outfile)
        else:
            raise FileNotFoundError(f"{json_path} does not exist.")
        return self.tuner_args

    def _save_db_file(self, file_path_dict):
        """
        save the database file to the same original path, if the path does not exist, create one.

        Args:
            file_path_dict (dict): dict which contains the path of different files
        """
        storage_dest_folder = os.path.dirname(
            self.tuner_args["storage"].replace("sqlite:///", "")
        )
        if not os.path.exists(storage_dest_folder):
            logger.debug(f"{self.storage_dest_folder} does not exist, creating one.")
            os.path.makedirs(storage_dest_folder)
        if os.path.exists(file_path_dict["db_path"]):
            shutil.copy(file_path_dict["db_path"], storage_dest_folder)
        else:
            raise FileNotFoundError(f'{file_path_dict["db_path"]} does not exist.')

    def deserialize_tuner_args(self, file_path_dict):
        """
        deserialize the tuner args

        Args:
            file_path_dict (dict): dict which contains the path of different files
        """
        self._does_model_exist()
        tuner_input = {}
        for field, value in self.tuner_args.items():
            if field == "model":
                tuner_input["model"] = self._deserialize_model(file_path_dict)
            elif field == "scoring":
                tuner_input["scoring"] = self._deserialize_scoring(file_path_dict)
            elif field == "strategy":
                tuner_input["strategy"] = decode(self.tuner_args["strategy"])
            elif field in ["loglevel", "storage", "study_name"]:
                tuner_input[field] = value
        return tuner_input

    def _does_model_exist(self):
        assert "model" in self.tuner_args, "`model` not found."

    def _deserialize_model(self, file_path_dict):
        model_args = self.tuner_args["model"]
        model_dir = os.path.dirname(file_path_dict["json_path"])
        if "model_name" in model_args:
            model = get_supported_model_mappings()[model_args["model_name"]](
                **model_args["model_input"]
            )
        elif "pickled_model_path" in model_args:
            model_path = os.path.join(model_dir, os.path.basename("model.pkl"))
            model = pickle.load(open(model_path, "rb"))
        elif "model_script" in model_args:
            model_path = os.path.join(
                model_dir, os.path.basename(model_args["model_script"])
            )
            model = self.load_model(model_path)
        else:
            raise RuntimeError("model cannot be found.")
        return model

    def _deserialize_scoring(self, file_path_dict):
        scoring_args = self.tuner_args["scoring"]
        if "scoring_name" in scoring_args:
            scoring = scoring_args["scoring_name"]
        elif "scoring_script" in scoring_args:
            scoring_path = os.path.join(
                os.path.dirname(file_path_dict["json_path"]),
                os.path.basename(scoring_args["scoring_script"]),
            )
            scoring = self.load_scoring(scoring_path)
        else:
            raise RuntimeError("scoring cannot be found.")
        return scoring

    @staticmethod
    def load_target_from_script(script_path):
        script_name = os.path.basename(script_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(
            script_name + "%s" % uuid.uuid4(), script_path
        )
        func = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(func)
        return func

    @staticmethod
    def load_model(script_path):
        spec = DownloadTunerArtifact.load_target_from_script(script_path)
        if hasattr(spec, "model"):
            model = spec.model
        else:
            raise Exception(
                "Need to explicitly define `model=your_model` in the model script."
            )
        return model

    @staticmethod
    def load_scoring(script_path):
        spec = DownloadTunerArtifact.load_target_from_script(script_path)
        if hasattr(spec, "scoring"):
            scoring = spec.scoring
        else:
            raise Exception(
                "Need to explicitly define `scoring=your_scoring_func` in the model script."
            )
        return scoring


@runtime_dependency(module="lightgbm", install_from=OptionalDependency.BOOSTED)
@runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
def get_supported_model_mappings():

    supported_model_mapping = {
        "Ridge": sklearn.linear_model._ridge.Ridge,
        "RidgeClassifier": sklearn.linear_model._ridge.RidgeClassifier,
        "Lasso": sklearn.linear_model._coordinate_descent.Lasso,
        "ElasticNet": sklearn.linear_model._coordinate_descent.ElasticNet,
        "LogisticRegression": sklearn.linear_model._logistic.LogisticRegression,
        "SVC": sklearn.svm._classes.SVC,
        "SVR": sklearn.svm._classes.SVR,
        "LinearSVC": sklearn.svm._classes.LinearSVC,
        "LinearSVR": sklearn.svm._classes.LinearSVR,
        "DecisionTreeClassifier": sklearn.tree._classes.DecisionTreeClassifier,
        "DecisionTreeRegressor": sklearn.tree._classes.DecisionTreeRegressor,
        "RandomForestClassifier": sklearn.ensemble._forest.RandomForestClassifier,
        "RandomForestRegressor": sklearn.ensemble._forest.RandomForestRegressor,
        "GradientBoostingClassifier": sklearn.ensemble._gb.GradientBoostingClassifier,
        "GradientBoostingRegressor": sklearn.ensemble._gb.GradientBoostingRegressor,
        "XGBClassifier": xgboost.sklearn.XGBClassifier,
        "XGBRegressor": xgboost.sklearn.XGBRegressor,
        "ExtraTreesClassifier": sklearn.ensemble._forest.ExtraTreesClassifier,
        "ExtraTreesRegressor": sklearn.ensemble._forest.ExtraTreesRegressor,
        "LGBMClassifier": lightgbm.sklearn.LGBMClassifier,
        "LGBMRegressor": lightgbm.sklearn.LGBMRegressor,
        "SGDClassifier": sklearn.linear_model._stochastic_gradient.SGDClassifier,
        "SGDRegressor": sklearn.linear_model._stochastic_gradient.SGDRegressor,
    }
    return supported_model_mapping
