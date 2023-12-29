from enum import Enum
from typing import Union, Optional, List

from ads.opctl.operator.lowcode.feature_store_marketplace.models.serializable_yaml_model import (
    SerializableYAMLModel,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.mysql_config import (
    MySqlConfig,
)


class APIGatewayConfig(SerializableYAMLModel):
    yaml_mapping = {
        "enabled": "enabled",
        "rootCompartmentId": "root_compartment_id",
        "region": "region",
        "authorizedUserGroups": "authorized_user_groups",
        "stackId": "stack_id",
    }

    def __init__(self):
        self.enabled = False
        self.root_compartment_id: Optional[str] = None
        self.region: Optional[str] = None
        self.authorized_user_groups: Optional[str] = None
        self.stack_id: Optional[str] = None
