#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Contains tests for ads.evaluations.evaluation_plot
"""

import pytest
import unittest

from ads.evaluations.evaluation_plot import EvaluationPlot
from ads.common import logger

#
# run with:
#  python -m pytest -v -p no:warnings --cov-config=.coveragerc --cov=./ --cov-report html /home/datascience/advanced-ds/tests/unitary/test_evaluations_evaluation_plot.py
#


class EvaluationPlotTest:
    """
    Contains test cases for ads.evaluations.evaluation_plot
    """

    def test_get_legend_labels(self):
        """
        Test get_legend_labels with proper parameter provided
        legend_labels has to be a dict with keys out of list on classes names
        """
        legend_labels = {"class_0": "one", "class_1": "two", "class_2": "three"}
        EvaluationPlot.classes = ["class_0", "class_1", "class_2"]
        EvaluationPlot.get_legend_labels(legend_labels)

        assert EvaluationPlot.legend_labels == legend_labels
