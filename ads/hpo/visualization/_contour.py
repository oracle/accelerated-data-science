#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from optuna.study import Study
from optuna.visualization.matplotlib import plot_contour


def _get_contour_plot(
    study: Study, fig_size: tuple, params: Optional[List[str]] = None
):
    plot_contour(study=study, params=params)
    plt.show()
