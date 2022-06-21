#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


@runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
def _get_contour_plot(
    study: "optuna.study.Study", fig_size: tuple, params: Optional[List[str]] = None
):
    from optuna.visualization.matplotlib import plot_contour

    plot_contour(study=study, params=params)
    plt.show(block=False)
