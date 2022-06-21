#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class NTrials:
    """
    Exit based on number of trials.

    Parameters
    ----------
    n_trials: int
        Number of trials (sets of hyperparamters tested). If :obj:`None`, there is no
        limitation on the number of trials.

    Returns
    -------
        :class:`NTrials`
            NTrials object
    """

    def __init__(self, n_trials: int):
        self.n_trials = n_trials

    def __call__(self):
        return self.n_trials


class TimeBudget:
    """
    Exit based on the number of seconds.

    Parameters
    ----------
    seconds: float
        Time limit, in seconds. If :obj:`None` there is no time limit.

    Returns
    -------
        :class:`TimeBudget`
            TimeBudget object
    """

    def __init__(self, seconds: float):
        assert seconds > 0, "<code>time_budget</code> has to be greater than 0."
        self.seconds = seconds

    def __call__(self):
        return self.seconds


class ScoreValue:
    """
    Exit if the score is greater than or equal to the threshold.

    Parameters
    ----------
    score: float
        The threshold for exiting the tuning process. If a trial value is greater or equal
        to `score`, process exits.

    Returns
    -------
        :class:`ScoreValue`
            ScoreValue object
    """

    def __init__(self, score: float):
        self.score = score

    def __call__(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"):
        if trial.value >= self.score:
            study.stop()
