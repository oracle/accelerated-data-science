#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import matplotlib.pyplot as plt
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


@runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
def _get_edf_plot(study: "optuna.study.Study", fig_size: tuple):

    from optuna.visualization.matplotlib import plot_edf

    plot_edf(study)
    plt.show(block=False)
