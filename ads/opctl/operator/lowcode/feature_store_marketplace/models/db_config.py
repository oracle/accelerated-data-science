#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Union, Optional

from ads.common.extended_enum import ExtendedEnum
from ads.opctl.operator.lowcode.feature_store_marketplace.models.atp_config import ATPConfig
from ads.opctl.operator.lowcode.feature_store_marketplace.models.serializable_yaml_model import (
    SerializableYAMLModel,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.mysql_config import (
    MySqlConfig,
)


class DBConfig(SerializableYAMLModel):
    yaml_mapping = {"configuredDB": "configured_db", "mysql": "mysql_config", "atp": "atp_config"}

    class DBType(ExtendedEnum):
        MySQL = "MYSQL"
        ATP = "ATP"

    def __init__(self):
        self._configured_db: Optional[DBConfig.DBType] = self.DBType.MySQL
        self._mysql_config: Optional[MySqlConfig] = None
        self._atp_config: Optional[ATPConfig] = None

    @property
    def configured_db(self) -> DBType:
        return self._configured_db

    @configured_db.setter
    def configured_db(self, configured_db: DBType):
        self._configured_db = configured_db

    @property
    def mysql_config(self) -> MySqlConfig:
        return self._mysql_config

    @mysql_config.setter
    def mysql_config(self, mysql_config: MySqlConfig):
        self._mysql_config = mysql_config

    @property
    def atp_config(self) -> ATPConfig:
        return self._atp_config

    @atp_config.setter
    def atp_config(self, atp_config: ATPConfig):
        self._atp_config = atp_config
