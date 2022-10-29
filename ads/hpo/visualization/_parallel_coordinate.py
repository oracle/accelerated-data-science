#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from collections import defaultdict
from typing import Callable, DefaultDict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

try:
    from optuna.visualization.matplotlib._matplotlib_imports import _imports
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `optuna` module was not found. Please run `pip install "
        f"{OptionalDependency.OPTUNA}`."
    )
except Exception as e:
    raise

if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import (
        Axes,
        LineCollection,
        plt,
    )


@runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
def plot_parallel_coordinate(
    study: "optuna.study.Study",
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[["optuna.trial.FrozenTrial"], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the high-dimensional parameter relationships in a study with Matplotlib."""

    _imports.check()
    optuna.visualization._utils._check_plot_args(study, target, target_name)
    return _get_parallel_coordinate_plot(study, params, target, target_name)


@runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
def _get_parallel_coordinate_plot(
    study: "optuna.study.Study",
    params: Optional[List[str]] = None,
    target: Optional[Callable[["optuna.trial.FrozenTrial"], float]] = None,
    target_name: str = "Objective Value",
    fig_size=None,
) -> "Axes":

    if target is None:

        def _target(t: "optuna.trial.FrozenTrial") -> float:
            return cast(float, t.value)

        target = _target
        reversescale = (
            study.direction == optuna.study._study_direction.StudyDirection.MINIMIZE
        )
    else:
        reversescale = True

    # Set up the graph style.
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues_r" if reversescale else "Blues")
    ax.set_title("Parallel Coordinate Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Prepare data for plotting.
    trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if len(trials) == 0:
        logger.warning("Your study does not have any completed trials.")
        return ax

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError(
                    "Parameter {} does not exist in your study.".format(input_p_name)
                )
        all_params = set(params)
    sorted_params = sorted(all_params)

    obj_org = [target(t) for t in trials]
    obj_min = min(obj_org)
    obj_max = max(obj_org)
    obj_w = obj_max - obj_min
    dims_obj_base = [[o] for o in obj_org]

    cat_param_names = []
    cat_param_values = []
    cat_param_ticks = []
    param_values = []
    var_names = [target_name]
    for p_name in sorted_params:
        values = [t.params[p_name] if p_name in t.params else np.nan for t in trials]

        from optuna.visualization.matplotlib._utils import _is_categorical

        if _is_categorical(trials, p_name):
            vocab = defaultdict(lambda: len(vocab))  # type: DefaultDict[str, int]
            values = [vocab[v] for v in values]
            cat_param_names.append(p_name)
            vocab_item_sorted = sorted(vocab.items(), key=lambda x: x[1])
            cat_param_values.append([v[0] for v in vocab_item_sorted])
            cat_param_ticks.append([v[1] for v in vocab_item_sorted])

        p_min = min(values)
        p_max = max(values)
        p_w = p_max - p_min

        if p_w == 0.0:
            center = obj_w / 2 + obj_min
            for i in range(len(values)):
                dims_obj_base[i].append(center)
        else:
            for i, v in enumerate(values):
                dims_obj_base[i].append((v - p_min) / p_w * obj_w + obj_min)

        var_names.append(p_name if len(p_name) < 20 else "{}...".format(p_name[:17]))
        param_values.append(values)

    # Draw multiple line plots and axes.
    # Ref: https://stackoverflow.com/a/50029441
    ax.set_xlim(0, len(sorted_params))
    ax.set_ylim(obj_min, obj_max)
    xs = [range(len(sorted_params) + 1) for _ in range(len(dims_obj_base))]
    segments = [np.column_stack([x, y]) for x, y in zip(xs, dims_obj_base)]
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(np.asarray([target(t) for t in trials] + [0]))
    axcb = fig.colorbar(lc, pad=0.1, ax=fig.gca())

    axcb.set_label(target_name)
    plt.xticks(range(len(sorted_params) + 1), var_names, rotation=330)

    for i, p_name in enumerate(sorted_params):
        ax2 = ax.twinx()
        ax2.set_ylim(min(param_values[i]), max(param_values[i]))

        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.plot([1] * len(param_values[i]), param_values[i], visible=False)
        ax2.spines["right"].set_position(("axes", (i + 1) / len(sorted_params)))
        if p_name in cat_param_names:
            idx = cat_param_names.index(p_name)
            tick_pos = cat_param_ticks[idx]
            tick_labels = cat_param_values[idx]
            ax2.set_yticks(tick_pos)
            ax2.set_yticklabels(tick_labels)

    ax.add_collection(lc)
    plt.show(block=False)
