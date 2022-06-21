#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional

import matplotlib.pyplot as plt
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


@runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
def _plot_param_importances(
    study: "optuna.study.Study",
    fig_size: tuple,
    evaluator: Optional["optuna.importance._base.BaseImportanceEvaluator"] = None,
    params: Optional[List[str]] = None,
):
    from optuna.visualization.matplotlib import plot_param_importances

    plot_param_importances(study, evaluator, params)
    plt.show(block=False)
