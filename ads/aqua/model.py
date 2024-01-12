#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List
from ads.model.datascience_model import DataScienceModel

logger = logging.getLogger(__name__)


class AquaModel(DataScienceModel):
    """Represents an Aqua Model.

    Attributes
    ----------

    Methods
    -------
    create(self, **kwargs) -> "AquaModel"
        Creates an instance of Aqua model.
    deploy(..., **kwargs)
        Deploys an Aqua model.
    list(self, ..., **kwargs) -> List["AquaModel"]
        List existing models created via Aqua

    """

    def __init__(self, **kwargs):
        """Initializes an Aqua model."""
        super().__init__(**kwargs)

    def create(self, **kwargs) -> "AquaModel":
        pass

    def deploy(self, **kwargs) -> "AquaModel":
        pass

    @classmethod
    def list(
        cls, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List["AquaModel"]:
        """List Aqua models in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        List[AquaModel]
            The list of the Aqua models.
        """

        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})
        all_models = cls.list(kwargs)
        # TODO: filter by free-form tags
        aqua_models = [cls()._update_from_oci_dsc_model(model) for model in all_models]
        return kwargs
