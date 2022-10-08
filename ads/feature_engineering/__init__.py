#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from ads.feature_engineering.feature_type.handler import warnings as feature_warning
from ads.feature_engineering.feature_type_manager import (
    FeatureTypeManager as feature_type_manager,
)
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.feature_type.base import Tag
from ads.feature_engineering.adsimage.image import ADSImage
from ads.feature_engineering.adsimage.image_reader import ADSImageReader
from ads.feature_engineering.feature_type.adsstring.oci_language import OCILanguage

feature_type = feature_type_manager.feature_type_registered

logger = logging.getLogger(__name__)
