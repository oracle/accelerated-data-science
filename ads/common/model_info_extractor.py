#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import abc
from ads.common.model_metadata import MetadataTaxonomyKeys


class ModelInfoExtractor(abc.ABC):
    """The base abstract class to extract model metadata.

    Methods
    -------
    framework(self) -> str
        Returns the framework of the model.
    algorithm(self) -> object
        Returns the algorithm of the model.
    version(self) -> str
        Returns the version of framework of the model.
    hyperparameter(self) -> dict
        Returns the hyperparameter of the model.
    info(self) -> dict
        Returns the model taxonomy metadata information.
    """

    @abc.abstractmethod
    def framework(self):
        """The abstract method to extracts the framework of the model.

        Returns
        ----------
        str:
           The framework of the model.
        """
        pass

    @abc.abstractmethod
    def algorithm(self):
        """The abstract method to extracts the algorithm of the model.

        Returns
        ----------
        object:
           The algorithm of the model.
        """
        pass

    @abc.abstractmethod
    def version(self):
        """The abstract method to extracts the framework version of the model.

        Returns
        ----------
        str:
           The framework version of the model.
        """
        pass

    @abc.abstractmethod
    def hyperparameter(self):
        """The abstract method to extracts the hyperparameters of the model.

        Returns
        ----------
        dict:
           The hyperparameter of the model.
        """
        pass

    def info(self):
        """Extracts the taxonomy metadata of the model.

        Returns
        ----------
        dict:
           The taxonomy metadata of the model.
        """
        return {
            MetadataTaxonomyKeys.FRAMEWORK: self.framework(),
            MetadataTaxonomyKeys.FRAMEWORK_VERSION: self.version(),
            MetadataTaxonomyKeys.ALGORITHM: str(self.algorithm()),
            MetadataTaxonomyKeys.HYPERPARAMETERS: self.hyperparameter(),
        }
