#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import matplotlib.pyplot as plt
from optuna.study import Study
from optuna.visualization.matplotlib import plot_optimization_history


def _get_optimization_history_plot(
    study: Study,
    fig_size: tuple,
    inferior: bool,
    best: bool,
):
    plot_optimization_history(study)
    plt.show()
