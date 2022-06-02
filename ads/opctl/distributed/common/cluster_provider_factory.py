#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class ClusterProviderFactory:
    """
    Factory class for creating provider instance.
    """

    provider = {}

    @classmethod
    def register(cls, cluster_type, provider_class):
        ClusterProviderFactory.provider[cluster_type.upper()] = provider_class

    @staticmethod
    def get_provider(key, *args, **kwargs):
        return ClusterProviderFactory.provider[key](*args, **kwargs)
