#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import matplotlib.pyplot as plt
from optuna.study import Study
from optuna.visualization.matplotlib import plot_intermediate_values


def _get_intermediate_plot(study: Study, fig_size: tuple):

    plot_intermediate_values(study)
    plt.show()
