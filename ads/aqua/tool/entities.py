#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from datetime import datetime
from typing import Literal, Optional

from pydantic import AnyUrl, ConfigDict

from ads.aqua.config.utils.serializer import Serializable


class AquaTool(Serializable):
    application_id: Optional[str] = None
    compartment_id: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None
    image: Optional[str] = None
    image_digest: Optional[str] = None
    invoke_endpoint: Optional[AnyUrl] = None
    lifecycle_tate: Optional[Literal["ACTIVE", "INACTIVE", "DELETING", "DELETED"]]
    memory_in_mbs: Optional[int] = None
    shape: Optional[str] = None
    time_created: Optional[datetime] = None
    time_updated: Optional[datetime] = None
    timeout_in_seconds: Optional[int] = None

    model_config = ConfigDict(extra="ignore")
