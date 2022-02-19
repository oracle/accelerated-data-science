#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.config.base import ConfigProcessor


class ConfigValidator(ConfigProcessor):
    def process(self):
        # TODO: add validation using pydantic + datamodel-code-generator in future PR

        # called to update command, part of which is encoded spec
        # and spec might have been updated during the call above
        # self.["execution"]["command"] = _resolve_command(self.config)
        return self
