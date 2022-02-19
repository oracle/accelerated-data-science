#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import abc

import six


@six.add_metaclass(abc.ABCMeta)
class AbstractTypeDiscoveryDetector:
    @abc.abstractmethod
    def discover(self, name, series):
        return


@six.add_metaclass(abc.ABCMeta)
class DiscreteDiscoveryDetector(AbstractTypeDiscoveryDetector):
    @abc.abstractmethod
    def discover(self, name, series):
        return
