#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
import os
import sys
import textwrap
from typing import List, Tuple, Union

import git
import numpy as np
import pandas as pd
from ads.common import logger, utils
from ads.common.data import ADSData
from ads.model.model_metadata import (
    MetadataCustomCategory,
    MetadataCustomKeys,
    MetadataTaxonomyKeys,
    ModelCustomMetadataItem,
    ModelProvenanceMetadata,
    UseCaseType,
)
from ads.common.utils import DATA_SCHEMA_MAX_COL_NUM, get_files
from ads.feature_engineering.schema import DataSizeTooWide, Schema, SchemaSizeTooLarge

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ADS")

METADATA_SIZE_LIMIT = 32000
METADATA_VALUE_LENGTH_LIMIT = 255
METADATA_DESCRIPTION_LENGTH_LIMIT = 255

MODEL_ARTIFACT_VERSION = "3.0"
INPUT_SCHEMA_FILE_NAME = "input_schema.json"
OUTPUT_SCHEMA_FILE_NAME = "output_schema.json"
ADS_PACKAGE_NAME = "ADS"


class MetadataMixin:
    """MetadataMixin class which populates the custom metadata, taxonomy metadata,
    input/output schema and provenance metadata.
    """

    def _populate_metadata_taxonomy(
        self, model: callable = None, use_case_type: str = None, **kwargs
    ):
        """Populates the taxonomy metadata.

        Parameters
        ----------
        model: (callable, optional). Defaults to None.
            Any model.
        use_case_type: (str, optional). Defaults to None.
            The use case type of the model.

        Raises
        ------
        ValueError: if `UseCaseType` is not from UseCaseType.values().

        Returns
        -------
        None
            Nothing.
        """
        if use_case_type and use_case_type not in UseCaseType:
            raise ValueError(
                f"Invalid value of `UseCaseType`. Choose from {UseCaseType.values()}."
            )

        self.metadata_taxonomy[MetadataTaxonomyKeys.USE_CASE_TYPE].value = use_case_type
        if model is not None:
            mapping = {
                MetadataTaxonomyKeys.FRAMEWORK: self.framework,
                MetadataTaxonomyKeys.FRAMEWORK_VERSION: self.version,
                MetadataTaxonomyKeys.ALGORITHM: str(self.algorithm),
                MetadataTaxonomyKeys.HYPERPARAMETERS: self.hyperparameter,
            }

            if mapping is not None:
                self.metadata_taxonomy._populate_from_map(mapping)
            if (
                self.metadata_taxonomy[MetadataTaxonomyKeys.HYPERPARAMETERS].size()
                > METADATA_SIZE_LIMIT
            ):
                logger.warn(
                    f"The model hyperparameters are larger than `{METADATA_SIZE_LIMIT}` "
                    "bytes and cannot be stored as model catalog metadata. It will be saved to "
                    f"{self.artifact_dir}/hyperparameters.json and removed from the metadata."
                )

                self.metadata_taxonomy[
                    MetadataTaxonomyKeys.HYPERPARAMETERS
                ].to_json_file(self.artifact_dir)
                self.metadata_taxonomy[MetadataTaxonomyKeys.HYPERPARAMETERS].update(
                    value=None
                )

    def _populate_metadata_custom(self):
        """Populates the custom metadata.

        Returns
        -------
        None
            Nothing.
        """
        model_metadata_items = []

        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.CONDA_ENVIRONMENT,
                value=self.runtime_info.model_provenance.training_conda_env.training_env_path,
                description="The conda environment where the model was trained.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )

        try:
            env_type = (
                self.runtime_info.model_provenance.training_conda_env.training_env_type
            )
        except:
            env_type = None
        try:
            slug_name = (
                self.runtime_info.model_provenance.training_conda_env.training_env_slug
            )
        except:
            slug_name = None
        try:
            env_path = (
                self.runtime_info.model_provenance.training_conda_env.training_env_path
            )
        except:
            env_path = None

        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.ENVIRONMENT_TYPE,
                value=env_type,
                description="The conda environment type, can be published or datascience.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.SLUG_NAME,
                value=slug_name,
                description="The slug name of the training conda environment.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.CONDA_ENVIRONMENT_PATH,
                value=env_path,
                description="The URI of the training conda environment.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )
        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.MODEL_ARTIFACTS,
                value=textwrap.shorten(
                    ", ".join(get_files(self.artifact_dir)),
                    255,
                    placeholder="...",
                ),
                description="The list of files located in artifacts folder.",
                category=MetadataCustomCategory.TRAINING_ENV,
            )
        )

        if self.model_file_name:
            model_metadata_items.append(
                ModelCustomMetadataItem(
                    key=MetadataCustomKeys.MODEL_SERIALIZATION_FORMAT,
                    value=self.model_file_name.split(".")[-1],
                    description="The model serialization format.",
                    category=MetadataCustomCategory.TRAINING_PROFILE,
                )
            )
            model_metadata_items.append(
                ModelCustomMetadataItem(
                    key=MetadataCustomKeys.MODEL_FILE_NAME,
                    value=self.model_file_name,
                    description="The model file name.",
                    category=MetadataCustomCategory.OTHER,
                )
            )
        else:
            logger.warning(
                "Unable to extract a model serialization format. "
                "The `model_file_name` is not provided."
            )

        model_metadata_items.append(
            ModelCustomMetadataItem(
                key=MetadataCustomKeys.CLIENT_LIBRARY,
                value=ADS_PACKAGE_NAME,
                description="",
                category=MetadataCustomCategory.OTHER,
            )
        )
        self.metadata_custom._add_many(model_metadata_items, replace=True)

    def _prepare_data_for_schema(
        self,
        X_sample: Union[List, Tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[List, Tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        **kwargs,
    ):
        """
        Any Framework-specific work before generic schema generation.
        """
        return X_sample, y_sample

    def _populate_provenance_metadata(
        self,
        training_script_path: str = "",
        training_id: str = None,
        ignore_pending_changes: bool = False,
    ):
        """Populates the custom metadata.

        Parameters
        ----------
        training_script_path: (str, optional). Defaults to "".
            Training script path.
        training_id: (str, optional). Defaults to None.
            The training model OCID.
        ignore_pending_changes: (bool, optional). Defaults to False.
            Ignore the pending changes in git.

        Returns
        -------
        None
            Nothing.
        """
        try:
            self.metadata_provenance = (
                ModelProvenanceMetadata.fetch_training_code_details(
                    training_script_path=training_script_path,
                    training_id=training_id,
                    artifact_dir=self.artifact_dir,
                )
            )
        except git.InvalidGitRepositoryError:
            self.metadata_provenance = ModelProvenanceMetadata()

        if training_script_path is not None:
            # training_script_path could be a directory or a path.
            # but if the path has too many files, it might take a long time.
            self.metadata_provenance.assert_path_not_dirty(
                training_script_path, ignore_pending_changes
            )
        self.metadata_provenance.assert_path_not_dirty(
            self.artifact_dir, ignore_pending_changes
        )

    def populate_metadata(
        self,
        use_case_type: str = None,
        data_sample: ADSData = None,
        X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        training_script_path: str = None,
        training_id: str = None,
        ignore_pending_changes: bool = True,
        max_col_num: int = DATA_SCHEMA_MAX_COL_NUM,
        ignore_conda_error: bool = False,
        **kwargs,
    ):
        """Populates input schema and output schema.
        If the schema exceeds the limit of 32kb, save as json files to the artifact directory.

        Parameters
        ----------
        use_case_type: (str, optional). Defaults to None.
            The use case type of the model.
        data_sample: (ADSData, optional). Defaults to None.
            A sample of the data that will be used to generate intput_schema and output_schema.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema.
        y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of output data that will be used to generate output schema.
        training_script_path: str. Defaults to None.
            Training script path.
        training_id: (str, optional). Defaults to None.
            The training model OCID.
        ignore_pending_changes: bool. Defaults to False.
            Ignore the pending changes in git.
        max_col_num: (int, optional). Defaults to utils.DATA_SCHEMA_MAX_COL_NUM.
            The maximum number of columns allowed in auto generated schema.

        Returns
        -------
        None
            Nothing.
        """
        if (
            self.estimator is None
            and self.metadata_taxonomy[MetadataTaxonomyKeys.ALGORITHM].value is None
        ):
            logger.info(
                "To auto-extract taxonomy metadata the model must be provided. Supported models: keras, lightgbm, pytorch, sklearn, tensorflow, pyspark, and xgboost."
            )
        if use_case_type is None:
            use_case_type = self.metadata_taxonomy[
                MetadataTaxonomyKeys.USE_CASE_TYPE
            ].value

        self._populate_metadata_taxonomy(
            model=self.estimator, use_case_type=use_case_type, **kwargs
        )
        if not ignore_conda_error:
            self._populate_metadata_custom()
        self.populate_schema(
            data_sample=data_sample,
            X_sample=X_sample,
            y_sample=y_sample,
            max_col_num=max_col_num,
            **kwargs,
        )
        self._populate_provenance_metadata(
            training_script_path=training_script_path,
            ignore_pending_changes=ignore_pending_changes,
            training_id=training_id,
        )
        self._summary_status.update_action(
            detail="Populated metadata(Custom, Taxonomy and Provenance)",
            action="",
        )
        self._summary_status.update_status(
            detail="Populated metadata(Custom, Taxonomy and Provenance)",
            status="Done",
        )

    def populate_schema(
        self,
        data_sample: ADSData = None,
        X_sample: Union[List, Tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[List, Tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        max_col_num: int = DATA_SCHEMA_MAX_COL_NUM,
        **kwargs,
    ):
        """Populate input and output schemas.
        If the schema exceeds the limit of 32kb, save as json files to the artifact dir.

        Parameters
        ----------
        data_sample: ADSData
            A sample of the data that will be used to generate input_schema and output_schema.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
            A sample of input data that will be used to generate the input schema.
        y_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
            A sample of output data that will be used to generate the output schema.
        max_col_num: (int, optional). Defaults to utils.DATA_SCHEMA_MAX_COL_NUM.
            The maximum number of columns allowed in auto generated schema.
        """
        if data_sample is not None:
            assert isinstance(
                data_sample, ADSData
            ), "`data_sample` expects data of ADSData type. \
            Pass in to `X_sample` and `y_sample` for other data types."
            X_sample = data_sample.X
            y_sample = data_sample.y

        X_sample, y_sample = self._prepare_data_for_schema(X_sample, y_sample, **kwargs)
        self.schema_input = self._populate_schema(
            X_sample,
            schema_file_name=INPUT_SCHEMA_FILE_NAME,
            max_col_num=max_col_num,
            **kwargs,
        )
        self.schema_output = self._populate_schema(
            y_sample,
            schema_file_name=OUTPUT_SCHEMA_FILE_NAME,
            max_col_num=max_col_num,
            **kwargs,
        )

    def _populate_schema(
        self,
        data: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame],
        schema_file_name: str,
        max_col_num: int,
        **kwargs,
    ):
        """Populates schema and if the schema exceeds the limit of 32kb, save as a json file to the artifact_dir.

        Parameters
        ----------
        data: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]
            A sample of input data that will be used to generate input schema.
        schema_file_name: str
            Filename of the schema file.
        max_col_num : int
            The maximum number of columns allowed in auto generated schema.

        Returns
        -------
        Schema
            The schema.
        """
        result = None

        try:
            if data is not None:
                data = utils.to_dataframe(data)
                schema = data.ads.model_schema(max_col_num=max_col_num)
                schema.to_json_file(
                    file_path=os.path.join(self.artifact_dir, schema_file_name),
                    storage_options=kwargs.pop("auth", {}),
                )
                if self._validate_schema_size(schema, schema_file_name):
                    result = schema
        except DataSizeTooWide:
            logger.warning(
                f"The data has too many columns and "
                f"the maximum allowable number of columns is `{max_col_num}`. "
                "The schema was not auto generated. increase allowable number of columns."
            )
        except NotImplementedError:
            logger.warning(
                f"Cannot convert the data to pandas dataframe, hence "
                "the schema was not auto generated."
            )

        return result or Schema()

    def _validate_schema_size(self, schema: Schema, schema_file_name: str):
        """Validate the schema size.

        Parameters
        ----------
        schema: (Schema)
            input/output schema.
        schema_file_name: (str)
            Filename of schema.

        Returns
        -------
        bool
            Whether the size of the schema is less than 32kb or not.
        """
        result = False
        try:
            result = schema.validate_size()
        except SchemaSizeTooLarge:
            logger.warn(
                f"The {schema_file_name.replace('.json', '')} is larger than "
                f"`{METADATA_SIZE_LIMIT}` bytes and cannot be stored as model catalog metadata."
                f"It will be saved to {self.artifact_dir}/{schema_file_name}."
            )

        return result
