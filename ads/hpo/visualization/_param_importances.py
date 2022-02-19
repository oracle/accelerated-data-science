#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional

import matplotlib.pyplot as plt
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.visualization.matplotlib import plot_param_importances


def _plot_param_importances(
    study: Study,
    fig_size: tuple,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
):

    plot_param_importances(study, evaluator, params)
    plt.show()
