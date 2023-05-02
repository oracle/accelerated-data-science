#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import importlib
import multiprocessing
import os
import uuid
import psutil
from enum import Enum, auto
from time import time, sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging
from ads.common import logger
from ads.common import utils
from ads.common.data import ADSData
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.hpo._imports import try_import
from ads.hpo.ads_search_space import get_model2searchspace
from ads.hpo.distributions import *
from ads.hpo.objective import _Objective
from ads.hpo.stopping_criterion import NTrials, ScoreValue, TimeBudget
from ads.hpo.utils import _num_samples, _safe_indexing, _update_space_name
from ads.hpo.validation import (
    assert_is_estimator,
    assert_model_is_supported,
    assert_strategy_valid,
    assert_tuner_is_fitted,
    validate_fit_params,
    validate_pipeline,
    validate_search_space,
    validate_params_for_plot,
)


with try_import() as _imports:
    from sklearn.base import BaseEstimator, clone, is_classifier
    from sklearn.model_selection import BaseCrossValidator  # NOQA
    from sklearn.model_selection import check_cv, cross_validate
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.utils import check_random_state
    from sklearn.exceptions import NotFittedError

    try:
        from sklearn.metrics import check_scoring
    except:
        from sklearn.metrics.scorer import check_scoring


from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union  # NOQA


class State(Enum):
    INITIATED = auto()
    RUNNING = auto()
    HALTED = auto()
    TERMINATED = auto()
    COMPLETED = auto()


class InvalidStateTransition(Exception):   # pragma: no cover
    """
    `Invalid State Transition` is raised when an invalid transition request is made, such as calling
    halt without a running process.
    """

    pass


class ExitCriterionError(Exception):   # pragma: no cover
    """
    `ExitCriterionError` is raised when an attempt is made to check exit status for a different exit
    type than the tuner was initialized with. For example, if an HPO study has an exit criteria based
    on the number of trials and a request is made for the time remaining, which is a different exit
    criterion, an exception is raised.
    """

    pass


class DuplicatedStudyError(Exception):   # pragma: no cover
    """
    `DuplicatedStudyError` is raised when a new tuner process is created with a study name that
    already exists in storage.
    """


class NoRestartError(Exception):   # pragma: no cover
    """
    `NoRestartError` is raised when an attempt is made to check how many seconds have transpired since
    the HPO process was last resumed from a halt. This can happen if the process has been terminated
    or it was never halted and then resumed to begin with.
    """

    pass


class DataScienceObjective:
    """This class is to replace the previous lambda function to solve the problem that python does not allow pickle local function/lambda function."""

    def __init__(self, objective, X_res, y_res):
        self.objective = objective
        self.X_res = X_res
        self.y_res = y_res

    def __call__(self, trial):
        return self.objective(self.X_res, self.y_res, trial)


class ADSTuner(BaseEstimator):
    """
    Hyperparameter search with cross-validation.
    """

    _required_parameters = ["model"]

    @property
    def sklearn_steps(self):
        """
        Returns
        -------
            int
                Search space which corresponds to the best candidate parameter setting.
        """
        return _update_space_name(self.best_params, step_name=self._step_name)

    @property
    def best_index(self):
        """
        Returns
        -------
            int
                Index which corresponds to the best candidate parameter setting.
        """
        return self.trials["value"].idxmax()

    @property
    def best_params(self):
        """
        Returns
        -------
            Dict[str, Any]
                Parameters of the best trial.
        """
        self._check_is_fitted()
        return self._remove_step_name(self._study.best_params)

    @property
    def best_score(self):
        """
        Returns
        -------
            float
                Mean cross-validated score of the best estimator.
        """
        self._check_is_fitted()
        return self._study.best_value

    @property
    def score_remaining(self):
        """
        Returns
        -------
            float
                The difference between the best score and the optimal score.

        Raises
        ------
            :class:`ExitCriterionError`
                Error is raised if there is no score-based criteria for tuning.
        """
        if self._optimal_score is None:
            raise ExitCriterionError(
                "Tuner does not have a score-based exit condition."
            )
        else:
            return self._optimal_score - self.best_score

    @property
    def scoring_name(self):
        """
        Returns
        -------
            str
                Scoring name.
        """
        return self._extract_scoring_name()

    @property
    def n_trials(self):
        """
        Returns
        -------
            int
                Number of completed trials. Alias for `trial_count`.
        """
        self._check_is_fitted()
        return len(self.trials)

    # Alias for n_trials
    trial_count = n_trials

    @property
    def trials_remaining(self):
        """
        Returns
        -------
            int
                The number of trials remaining in the budget.

        Raises
        ------
            :class:`ExitCriterionError`
                Raised if the current tuner does not include a trials-based exit
                condition.
        """
        if self._n_trials is None:
            raise ExitCriterionError(
                "This tuner does not include a trials-based exit condition"
            )
        return self._n_trials - self.n_trials + self._previous_trial_count

    @property
    def trials(self):
        """
        Returns
        -------
            :class:`pandas.DataFrame`
                Trial data up to this point.
        """
        if self.is_halted():
            if self._trial_dataframe is None:
                return pd.DataFrame()
            return self._trial_dataframe
        trials_dataframe = self._study.trials_dataframe().copy()
        return trials_dataframe

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def __init__(
        self,
        model,  # type: Union[BaseEstimator, Pipeline]
        strategy="perfunctory",  # type: Union[str, Mapping[str, optuna.distributions.BaseDistribution]]
        scoring=None,  # type: Optional[Union[Callable[..., float], str]]
        cv=5,  # type: Optional[int]
        study_name=None,  # type: Optional[str]
        storage=None,  # type: Optional[str]
        load_if_exists=True,  # type: Optional[bool]
        random_state=None,  # type: Optional[int]
        loglevel=logging.INFO,  # type: Optional[int]
        n_jobs=1,  # type: Optional[int]
        X=None,  # type: Union[List[List[float]], np.ndarray, pd.DataFrame, spmatrix, ADSData]
        y=None,  # type: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]]
    ):
        # type: (...) -> None
        """
        Returns a hyperparameter tuning object

        Parameters
        ----------
            model:
                Object to use to fit the data. This is assumed to implement the
                scikit-learn estimator or pipeline interface.
            strategy:
                ``perfunctory``, ``detailed`` or a dictionary/mapping of hyperparameter
                and its distribution . If obj:`perfunctory`, picks a few
                relatively more important hyperparmeters to tune . If obj:`detailed`,
                extends to a larger search space. If obj:dict, user defined search
                space: Dictionary where keys are hyperparameters and values are distributions.
                Distributions are assumed to implement the ads distribution interface.
            scoring: Optional[Union[Callable[..., float], str]]
                String or callable to evaluate the predictions on the validation data.
                If :obj:`None`, ``score`` on the estimator is used.
            cv: int
                Integer to specify the number of folds in a CV splitter.
                If :obj:`estimator` is a classifier and :obj:`y` is
                either binary or multiclass,
                ``sklearn.model_selection.StratifiedKFold`` is used. otherwise,
                ``sklearn.model_selection.KFold`` is used.
            study_name: str,
                Name of the current experiment for the ADSTuner object. One ADSTuner
                object can only be attached to one study_name.
            storage:
                Database URL. (e.g. sqlite:///example.db). Default to sqlite:////tmp/hpo_*.db.
            load_if_exists:
                Flag to control the behavior to handle a conflict of study names.
                In the case where a study named ``study_name`` already exists in the ``storage``,
                a :class:`DuplicatedStudyError` is raised if ``load_if_exists`` is
                set to :obj:`False`.
                Otherwise, the existing one is returned.
            random_state:
                Seed of the pseudo random number generator. If int, this is the
                seed used by the random number generator. If :obj:`None`, the global random state from
                ``numpy.random`` is used.
            loglevel:
                loglevel. can be logging.NOTSET, logging.INFO, logging.DEBUG, logging.WARNING
            n_jobs: int
                Number of parallel jobs. :obj:`-1` means using all processors.
            X: TwoDimArrayLikeType, Union[List[List[float]], np.ndarray,
            pd.DataFrame, spmatrix, ADSData]
                Training data.
            y: Union[OneDimArrayLikeType, TwoDimArrayLikeType], optional
            OneDimArrayLikeType: Union[List[float], np.ndarray, pd.Series]
            TwoDimArrayLikeType: Union[List[List[float]], np.ndarray, pd.DataFrame, spmatrix, ADSData]
                Target.

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.svm import SVC

            tuner = ADSTuner(
                            SVC(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )

            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
        """
        _imports.check()
        self._n_jobs = n_jobs
        assert (
            cv > 1
        ), "k-fold cross-validation requires at least one train/test split by setting cv=2 or more"
        self.cv = cv
        self._error_score = np.nan
        self.model = model
        self._check_pipeline()
        self._step_name = None
        self._extract_estimator()
        self.strategy = None
        self._param_distributions = None
        self._check_strategy(strategy)
        self.strategy = strategy
        self._param_distributions = self._get_param_distributions(self.strategy)
        self._enable_pruning = hasattr(self.model, "partial_fit")
        self._max_iter = 100
        self.__random_state = random_state  # to be used in export_trials
        # this calls the randomstate.setter which turns self.random_state into a np.random.RandomState instance
        # make it hard to be serialized.
        self.random_state = check_random_state(random_state)

        self._return_train_score = False
        self.scoring = scoring
        self._subsample = 1.0
        self.loglevel = loglevel
        self._trial_dataframe = None
        self._status = State.INITIATED
        self.study_name = (
            study_name if study_name is not None else "hpo_" + str(uuid.uuid4())
        )
        self.storage = (
            "sqlite:////tmp/hpo_" + str(uuid.uuid4()) + ".db"
            if storage is None
            else storage
        )
        self.oci_client = None

        seed = np.random.randint(0, np.iinfo("int32").max)

        self.sampler = optuna.samplers.TPESampler(seed=seed)
        self.median_pruner = self._pruner(
            class_name="median_pruner",
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )
        self.load_if_exists = load_if_exists
        try:
            self._study = optuna.study.create_study(
                study_name=self.study_name,
                direction="maximize",
                pruner=self.median_pruner,
                sampler=self.sampler,
                storage=self.storage,
                load_if_exists=self.load_if_exists,
            )
        except optuna.exceptions.DuplicatedStudyError as e:
            if self.load_if_exists:
                logger.info(
                    "Using an existing study with name '{}' instead of "
                    "creating a new one.".format(self.study_name)
                )
            else:
                raise DuplicatedStudyError(
                    f"The study_name `{self.study_name}` exists in the {self.storage}. Either set load_if_exists=True, or use a new study_name."
                )
        self._init_data(X, y)

    def search_space(self, strategy=None, overwrite=False):
        """
        Returns the search space. If strategy is not passed in, return the existing search
        space. When strategy is passed in, overwrite the existing search space if overwrite
        is set True, otherwise, only update the existing search space.

        Parameters
        ----------
        strategy: Union[str, dict], optional
            ``perfunctory``, ``detailed`` or a dictionary/mapping of the hyperparameters
            and their distributions. If obj:`perfunctory`, picks a few relatively
            more important hyperparmeters to tune . If obj:`detailed`, extends to a
            larger search space. If obj:dict, user defined search space: Dictionary
            where keys are parameters and values are distributions. Distributions are
            assumed to implement the ads distribution interface.
        overwrite: bool, optional
            Ignored when strategy is None. Otherwise, search space is overwritten if overwrite
            is set True and updated if it is False.

        Returns
        -------
            dict
                A mapping of the hyperparameters and their distributions.

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
            tuner.search_space()
        """
        assert hasattr(
            self, "_param_distributions"
        ), "Call <code>ADSTuner</code> first."
        if not strategy:
            return self._remove_step_name(self._param_distributions)
        self._check_strategy(strategy)
        self.strategy = strategy
        if overwrite:
            self._param_distributions = self._get_param_distributions(self.strategy)
        else:
            self._param_distributions.update(
                self._get_param_distributions(self.strategy)
            )
        return self._remove_step_name(self._param_distributions)

    @staticmethod
    def _remove_step_name(param_distributions):
        search_space = {}
        for param, distributions in param_distributions.items():
            if "__" in param:
                param = param.split("__")[1]
            search_space[param] = distributions
        return search_space

    def _check_pipeline(self):
        self.model = validate_pipeline(self.model)

    def _get_internal_param_distributions(self, strategy):
        if isinstance(self.model, Pipeline):
            for step_name, step in self.model.steps:
                if step.__class__ in get_model2searchspace().keys():
                    self._step_name = step_name
                    param_distributions = get_model2searchspace()[step.__class__](
                        strategy
                    ).suggest_space(step_name=step_name)
            if len(param_distributions) == 0:
                logger.warning("Nothing to tune.")
        else:
            assert_model_is_supported(self.model)
            param_distributions = get_model2searchspace()[self.model.__class__](
                strategy
            ).suggest_space()
            self._check_search_space(param_distributions)
        return param_distributions

    def _get_param_distributions(self, strategy):
        if isinstance(strategy, str):
            param_distributions = self._get_internal_param_distributions(strategy)
        if isinstance(strategy, dict):
            param_distributions = _update_space_name(
                strategy, step_name=self._step_name
            )
            self._check_search_space(param_distributions)
        return param_distributions

    def _check_search_space(self, param_distributions):

        validate_search_space(self.model.get_params().keys(), param_distributions)

    def _check_is_fitted(self):
        assert_tuner_is_fitted(self)

    def _check_strategy(self, strategy):
        assert_strategy_valid(self._param_distributions, strategy, self.strategy)

    def _add_halt_time(self):
        """Adds a new start time window to the start/stop log. This happens in two cases: when the tuning process
        has commenced and when it resumes following a halt
        """
        self._time_log.append(dict(halt=time(), resume=None))

    def _add_resume_time(self):
        """Adds a new stopping time to the last window in the time log. This happens when the HPO process is
        halted or terminated.
        """
        if len(self._time_log) > 0:
            entry = self._time_log.pop()
        if entry["resume"] is not None:
            raise Exception("Cannot close a time window without an opening time.")
        self._time_log.append(dict(halt=entry["halt"], resume=time()))

    def tune(
        self,
        X=None,  # type: TwoDimArrayLikeType
        y=None,  # type: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]]
        exit_criterion=[],  # type: Optional[list]
        loglevel=None,  # type: Optional[int]
        synchronous=False,  # type: Optional[boolean]
    ):
        """
        Run hypyerparameter tuning until one of the <code>exit_criterion</code>
        is met. The default is to run 50 trials.

        Parameters
        ----------
        X: TwoDimArrayLikeType, Union[List[List[float]], np.ndarray, pd.DataFrame, spmatrix, ADSData]

            Training data.
        y: Union[OneDimArrayLikeType, TwoDimArrayLikeType], optional
        OneDimArrayLikeType: Union[List[float], np.ndarray, pd.Series]
        TwoDimArrayLikeType: Union[List[List[float]], np.ndarray, pd.DataFrame, spmatrix, ADSData]

            Target.
        exit_criterion: list, optional
            A list of ads stopping criterion. Can be `ScoreValue()`, `NTrials()`, `TimeBudget()`.
            For example, [ScoreValue(0.96), NTrials(40), TimeBudget(10)]. It will exit when any of the
            stopping criterion is satisfied in the `exit_criterion` list.
            By default, the run will stop after 50 trials.
        loglevel: int, optional
            Log level.
        synchronous: boolean, optional
            Tune synchronously or not. Defaults to `False`

        Returns
        -------
            None
                Nothing

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.svm import SVC

            tuner = ADSTuner(
                            SVC(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
        """

        # Get previous trial count to ensure proper counting.
        try:
            self._previous_trial_count = self.trial_count
        except NotFittedError:
            self._previous_trial_count = 0
        except Exception as e:
            _logger.error(f"Error retrieving previous trial count: {e}")
            raise

        self._init_data(X, y)
        if self.X is None:
            raise ValueError(
                "Need to either pass the data to `X` and `y` in `tune()`, or to `ADSTuner`."
            )
        if self.is_running():
            raise InvalidStateTransition(
                "Running process found. Do you need to call terminate() to stop before calling tune()?"
            )
        if self.is_halted():
            raise InvalidStateTransition(
                "Halted process found. You need to call resume()."
            )
        # handle ADSData

        # Initialize time log for every new call to tune(). Set shared global time values
        self._global_start = multiprocessing.Value("d", 0.0)
        self._global_stop = multiprocessing.Value("d", 0.0)
        self._time_log = []

        self._tune(
            X=self.X,
            y=self.y,
            exit_criterion=exit_criterion,
            loglevel=loglevel,
            synchronous=synchronous,
        )

        # Tune cannot exit before the clock starts in the subprocess.
        while self._global_start.value == 0.0:
            sleep(0.01)

    def _init_data(self, X, y):
        if X is not None:
            if isinstance(X, ADSData):
                self.y = X.y
                self.X = X.X
            else:
                self.X = X
                self.y = y

    def halt(self):
        """
        Halt the current running tuning process.

        Returns
        -------
            None
                Nothing

        Raises
        ------
            `InvalidStateTransition` if no running process is found

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
            tuner.halt()
        """
        if hasattr(self, "_tune_process") and self._status == State.RUNNING:
            self._trial_dataframe = self._study.trials_dataframe().copy()
            psutil.Process(self._tune_process.pid).suspend()
            self._status = State.HALTED
            self._add_halt_time()
        else:
            raise InvalidStateTransition(
                "No running process found. Do you need to call tune()?"
            )

    def resume(self):
        """
        Resume the current halted tuning process.

        Returns
        -------
            None
                Nothing

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
            tuner.halt()
            tuner.resume()
        """
        if self.is_halted():
            psutil.Process(self._tune_process.pid).resume()
            self._add_resume_time()
            self._status = State.RUNNING
        else:
            raise InvalidStateTransition("No paused process found.")

    def wait(self):
        """
        Wait for the current tuning process to finish running.

        Returns
        -------
            None
                Nothing

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
            tuner.wait()
        """
        if self.is_running():
            self._tune_process.join()
            self._status = State.COMPLETED
        else:
            raise InvalidStateTransition("No running process.")

    def terminate(self):
        """
        Terminate the current tuning process.

        Returns
        -------
            None
                Nothing

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
            tuner.terminate()
        """
        if self.is_running():
            self._tune_process.terminate()
            self._tune_process.join()
            self._status = State.TERMINATED
            # self._add_terminate_time()
            self._update_failed_trial_state()
        else:
            raise RuntimeError("No running process found. Do you need to call tune()?")

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def _update_failed_trial_state(self):
        from optuna.trial import TrialState

        for trial in self._study.trials:
            if trial.state == TrialState.RUNNING:
                self._study._storage.set_trial_state(
                    trial._trial_id, optuna.structs.TrialState.FAIL
                )

    @property
    def time_remaining(self):
        """Returns the number of seconds remaining in the study

        Returns
        -------
            int: Number of seconds remaining in the budget. 0 if complete/terminated

        Raises
        ------
            :class:`ExitCriterionError`
                Error is raised if time has not been included in the budget.
        """
        if self._time_budget is None:
            raise ExitCriterionError(
                "This tuner does not include a time-based exit condition"
            )
        elif self.is_completed() or self.is_terminated():
            return 0
        return max(self._time_budget - self.time_elapsed, 0)

    @property
    def time_since_resume(self):
        """Return the seconds since the process has been resumed from a halt.

        Returns
        -------
            int: the number of seconds since the process was last resumed

        Raises
        ------
            `NoRestartError` is the process has not been resumed

        """
        if len(self._time_log) > 0:
            last_time_resumed = self._time_log[-1].get("resume")
        else:
            raise Exception("Time log should not be empty")

        if self.is_running():
            if last_time_resumed is not None:
                return time() - last_time_resumed
            else:
                raise NoRestartError("The process has not been resumed")
        elif self.is_halted():
            return 0  # if halted, the amount of time since restarted from a halt is 0
        elif self.is_terminated():
            raise NoRestartError("The process has been terminated")

    @property
    def time_elapsed(self):
        """Return the time in seconds that the HPO process has been searching

        Returns
        -------
            int: The number of seconds the HPO process has been searching
        """
        time_in_halted_state = 0.0

        # Add up all the halted durations, i.e. the time spent between halt and resume
        for entry in self._time_log:
            halt_time = entry.get("halt")
            resume_time = entry.get("resume")

            if resume_time is None:
                # halted state.
                # elapsed = halt time - global start - time halted
                elapsed = halt_time - self._global_start.value - time_in_halted_state
                return elapsed

            else:
                # running/completed/terminated state,
                time_in_halted_state += resume_time - halt_time

        # If the loop ends all halts were resumed. If self._global_stop != 0 that means the
        # process has exited.
        if self._global_stop.value != 0:
            global_time = self._global_stop.value - self._global_start.value
        else:
            global_time = time() - self._global_start.value

        elapsed = global_time - time_in_halted_state
        return elapsed

    def best_scores(self, n: int = 5, reverse: bool = True):
        """Return the best scores from the study

        Parameters
        ----------
        n: int
            The maximum number of results to show. Defaults to 5. If `None` or
            negative return all.
        reverse: bool
            Whether to reverse the sort order so results are in descending order.
            Defaults to `True`

        Returns
        -------
        list[float or int]
            List of the best scores

        Raises
        ------
        `ValueError` if there are no trials
        """
        if len(self.trials) < 1:
            raise ValueError("No score data to show")
        else:
            scores = self.trials.value
            scores = scores[scores.notnull()]
            if scores is None:
                raise ValueError(
                    f"No score data despite valid trial data. Trial data length: {len(self.trials)}"
                )
            if not isinstance(n, int) or n <= 0:
                return sorted(scores, reverse=reverse)
            else:
                return sorted(scores, reverse=reverse)[:n]

    def get_status(self):
        """
        return the status of the current tuning process.

        Alias for the property `status`.

        Returns
        -------
            :class:`Status`
                The status of the process

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)])
            tuner.get_status()
        """
        return self.status

    def is_running(self):
        """
        Returns
        -------
            bool
                `True` if the :class:`ADSTuner` instance is running; `False` otherwise.
        """
        return self.status == State.RUNNING

    def is_halted(self):
        """
        Returns
        -------
            bool
                `True` if the :class:`ADSTuner` instance is halted; `False` otherwise.
        """
        return self.status == State.HALTED

    def is_terminated(self):
        """
        Returns
        -------
            bool
                `True` if the :class:`ADSTuner` instance has been terminated; `False` otherwise.
        """
        return self.status == State.TERMINATED

    def is_completed(self):
        """
        Returns
        -------
            bool
                `True` if the :class:`ADSTuner` instance has completed; `False` otherwise.
        """
        return self.status == State.COMPLETED

    def _is_tuning_started(self):
        """
        Returns
        -------
            bool
                `True` if the :class:`ADSTuner` instance has been started (for example, halted or
                running); `False` otherwise.
        """
        return self.status == State.HALTED or self.status == State.RUNNING

    def _is_tuning_finished(self):
        """
        Returns
        -------
            bool
                `True` if the :class:`ADSTuner` instance is finished running (i.e. completed
                or terminated); `False` otherwise.
        """
        return self.status == State.COMPLETED or self.status == State.TERMINATED

    @property
    def status(self):
        """
        Returns
        -------
            :class:`Status`
                The status of the current tuning process.
        """
        if (
            self._status == State.HALTED
            or self._status == State.TERMINATED
            or self._status == State.INITIATED
        ):
            return self._status
        elif hasattr(self, "_tune_process") and self._tune_process.is_alive():
            return State.RUNNING
        else:
            return State.COMPLETED
        return self._status

    def _extract_exit_criterion(self, exit_criterion):
        # handle the exit criterion
        self._time_budget = None
        self._n_trials = None
        self.exit_criterion = []
        self._optimal_score = None
        if exit_criterion is None or len(exit_criterion) == 0:
            self._n_trials = 50
        for i, criteria in enumerate(exit_criterion):
            if isinstance(criteria, TimeBudget):
                self._time_budget = criteria()
            elif isinstance(criteria, NTrials):
                self._n_trials = criteria()
            elif isinstance(criteria, ScoreValue):
                self._optimal_score = criteria.score
                self.exit_criterion.append(criteria)
            else:
                raise NotImplementedError(
                    "``{}`` is not supported!".format(criteria.__class__.__name__)
                )

    def _extract_estimator(self):
        if isinstance(self.model, Pipeline):  # Pipeline
            for step_name, step in self.model.steps:
                if self._is_estimator(step):
                    self._step_name = step_name
                    self.estimator = step

        else:
            self.estimator = self.model
            assert_is_estimator(self.estimator)
            # assert _check_estimator(self.estimator), "Estimator must implement fit"

    def _extract_scoring_name(self):
        if isinstance(self.scoring, str):
            return self.scoring
        if self._scorer.__class__.__name__ != "function":
            return (
                self._scorer
                if isinstance(self._scorer, str)
                else str(self._scorer).split("(")[1].split(")")[0]
            )
        else:
            if is_classifier(self.model):
                return "mean accuracy"
            else:
                return "r2"

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def _set_logger(self, loglevel, class_name):
        if loglevel is not None:
            self.loglevel = loglevel
        if class_name == "optuna":
            optuna.logging.set_verbosity(self.loglevel)
        else:
            raise NotImplementedError("{} is not supported.".format(class_name))

    def _set_sample_indices(self, X, random_state):
        max_samples = self._subsample
        n_samples = _num_samples(X)
        self._sample_indices = np.arange(n_samples)

        if isinstance(max_samples, float):
            max_samples = int(max_samples * n_samples)

        if max_samples < n_samples:
            self._sample_indices = random_state.choice(
                self._sample_indices, max_samples, replace=False
            )

            self._sample_indices.sort()

    def _get_fit_params_res(self, X):
        fit_params = {}
        fit_params_res = fit_params

        if fit_params_res is not None:
            fit_params_res = validate_fit_params(X, fit_params, self._sample_indices)
        return fit_params_res

    def _can_tune(self):
        assert hasattr(self, "model"), "Call <code>ADSTuner</code> first."
        if self._param_distributions == {}:
            logger.warning("Nothing to tune.")

        if self._param_distributions is None:
            raise NotImplementedError(
                "There was no model specified or the model is not supported."
            )

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def _tune(
        self,
        X,  # type: TwoDimArrayLikeType
        y,  # type: Optional[Union[OneDimArrayLikeType, TwoDimArrayLikeType]]
        exit_criterion=[],  # type: Optional[list]
        loglevel=None,  # type: Optional[int]
        synchronous=False,  # type: Optional[boolean]
    ):
        # type: (...) -> tuple
        """
        Tune with all sets of parameters.
        """
        self._can_tune()
        self._set_logger(loglevel=loglevel, class_name="optuna")
        self._extract_exit_criterion(exit_criterion)
        self._extract_estimator()
        random_state = self.random_state
        old_level = logger.getEffectiveLevel()
        logger.setLevel(self.loglevel)
        if not synchronous:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            logger.setLevel(logging.ERROR)

        self._set_sample_indices(X, random_state)
        X_res = _safe_indexing(X, self._sample_indices)
        y_res = _safe_indexing(y, self._sample_indices)
        groups_res = _safe_indexing(None, self._sample_indices)
        fit_params_res = self._get_fit_params_res(X)

        classifier = is_classifier(self.model)
        cv = check_cv(self.cv, y_res, classifier=classifier)
        self._n_splits = cv.get_n_splits(X_res, y_res, groups=groups_res)

        # scoring
        self._scorer = check_scoring(self.estimator, scoring=self.scoring)

        self._study = optuna.study.create_study(
            study_name=self.study_name,
            direction="maximize",
            pruner=self.median_pruner,
            sampler=self.sampler,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )
        objective = _Objective(
            self.model,
            self._param_distributions,
            cv,
            self._enable_pruning,
            self._error_score,
            fit_params_res,
            groups_res,
            self._max_iter,
            self._return_train_score,
            self._scorer,
            self.scoring_name,
            self._step_name,
        )

        if synchronous:
            logger.info(
                "Optimizing hyperparameters using {} "
                "samples...".format(_num_samples(self._sample_indices))
            )

        self._tune_process = multiprocessing.Process(
            target=ADSTuner.optimizer,
            args=(
                self.study_name,
                self.median_pruner,
                self.sampler,
                self.storage,
                self.load_if_exists,
                DataScienceObjective(objective, X_res, y_res),
                self._global_start,
                self._global_stop,
            ),
            kwargs=dict(
                n_jobs=self._n_jobs,
                n_trials=self._n_trials,
                timeout=self._time_budget,
                show_progress_bar=False,
                callbacks=self.exit_criterion,
                gc_after_trial=False,
            ),
        )

        self._tune_process.start()
        self._status = State.RUNNING

        if synchronous:
            self._tune_process.join()
            logger.info("Finished hyperparemeter search!")
            self._status = State.COMPLETED

        logger.setLevel(old_level)

    @staticmethod
    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def optimizer(
        study_name,
        pruner,
        sampler,
        storage,
        load_if_exists,
        objective_func,
        global_start,
        global_stop,
        **kwargs,
    ):
        """
        Static method for running ADSTuner tuning process

        Parameters
        ----------
            study_name: str
                The name of the study.
            pruner
                The pruning method for pruning trials.
            sampler
                The sampling method used for tuning.
            storage: str
                Storage endpoint.
            load_if_exists: bool
                Load existing study if it exists.
            objective_func
                The objective function to be maximized.
            global_start: :class:`multiprocesing.Value`
                The global start time.
            global_stop: :class:`multiprocessing.Value`
                The global stop time.
            kwargs: dict
                Keyword/value pairs passed into the optimize process


        Raises
        ------
            :class:`Exception`
                Raised for any exceptions thrown by the underlying optimization process

        Returns
        -------
            None
                Nothing

        """
        import traceback

        study = optuna.study.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            storage=storage,
            load_if_exists=load_if_exists,
        )
        try:
            global_start.value = time()
            study.optimize(objective_func, **kwargs)
            global_stop.value = time()
        except Exception as e:
            traceback.print_exc()
            raise e

    @staticmethod
    def _is_estimator(step):
        return hasattr(step, "fit") and (
            not hasattr(step, "transform")
            or hasattr(step, "predict")
            or hasattr(step, "fit_predict")
        )

    @staticmethod
    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def _pruner(class_name, **kwargs):
        if class_name == "median_pruner":
            return optuna.pruners.MedianPruner(**kwargs)
        else:
            raise NotImplementedError("{} is not supported.".format(class_name))

    def trials_export(
        self, file_uri, metadata=None, script_dict={"model": None, "scoring": None}
    ):
        """Export the meta data as well as files needed to reconstruct the ADSTuner object to the object storage.
        Data is not stored. To resume the same ADSTuner object from object storage and continue tuning from previous trials,
        you have to provide the dataset.

        Parameters
        ----------
            file_uri: str
                Object storage path, 'oci://bucketname@namespace/filepath/on/objectstorage'. For example,
                `oci://test_bucket@ociodsccust/tuner/test.zip`
            metadata: str, optional
                User defined metadata
            script_dict: dict, optional
                Script paths for model and scoring. This is only recommended for unsupported
                models and user-defined scoring functions. You can store the model and scoring
                function in a dictionary with keys `model` and `scoring` and the respective
                paths as values. The model and scoring scripts must import necessary libraries
                for the script to run. The ``model`` and ``scoring`` variables must be set to
                your model and scoring function.

        Returns
        -------
            None
                Nothing

        Example::

            # Print out a list of supported models
            from ads.hpo.ads_search_space import model_list
            print(model_list)

            # Example scoring dictionary
            {'model':'/home/datascience/advanced-ds/notebooks/scratch/ADSTunerV2/mymodel.py',
            'scoring':'/home/datascience/advanced-ds/notebooks/scratch/ADSTunerV2/customized_scoring.py'}

        Example::

            from ads.hpo.stopping_criterion import *
            from ads.hpo.search_cv import ADSTuner
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier

            tuner = ADSTuner(
                            SGDClassifier(),
                            strategy='detailed',
                            scoring='f1_weighted',
                            random_state=42
                        )
            tuner.search_space({'max_iter': 100})
            X, y = load_iris(return_X_y=True)
            tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)], synchronous=True)
            tuner.trials_export('oci://<bucket_name>@<namespace>/tuner/test.zip')
        """
        # oci://bucketname@namespace/filename
        from ads.hpo.tuner_artifact import UploadTunerArtifact

        assert self._is_tuning_finished()
        assert script_dict.keys() <= set(
            ["model", "scoring"]
        ), "script_dict keys can only be model and scoring."

        UploadTunerArtifact(self, file_uri, metadata).upload(script_dict)

    @classmethod
    def trials_import(cls, file_uri, delete_zip_file=True, target_file_path=None):
        """Import the database file from the object storage

        Parameters
        ----------
        file_uri: str
            'oci://bucketname@namespace/filepath/on/objectstorage'
            Example: 'oci://<bucket_name>@<namespace>/tuner/test.zip'
        delete_zip_file: bool, defaults to True, optional
            Whether delete the zip file afterwards.
        target_file_path: str, optional
            The path where the zip file will be saved. For example, '/home/datascience/myfile.zip'.

        Returns
        -------
        :class:`ADSTuner`
            ADSTuner object

        Examples
        --------
        >>> from ads.hpo.stopping_criterion import *
        >>> from ads.hpo.search_cv import ADSTuner
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import SGDClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> tuner = ADSTuner.trials_import('oci://<bucket_name>@<namespace>/tuner/test.zip')
        >>> tuner.tune(X=X, y=y, exit_criterion=[TimeBudget(1)], synchronous=True)
        """
        from ads.hpo.tuner_artifact import DownloadTunerArtifact

        tuner_args, cls.metadata = DownloadTunerArtifact(
            file_uri, target_file_path=target_file_path
        ).extract_tuner_args(delete_zip_file=delete_zip_file)
        return cls(**tuner_args)

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def _plot(
        self,  # type: ADSTuner
        plot_module,  # type: str
        plot_func,  # type: str
        time_interval=0.5,  # type: float
        fig_size=(800, 500),  # type: tuple
        **kwargs,
    ):
        if fig_size:
            logger.warning(
                "The param fig_size will be depreciated in future releases.",
            )

        spec = importlib.util.spec_from_file_location(
            "plot",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "visualization",
                plot_module + ".py",
            ),
        )
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)

        _imports.check()
        assert self._study is not None, "Need to call <code>.tune()</code> first."
        ntrials = 0
        if plot_func == "_plot_param_importances":
            print("Waiting for more trials before evaluating the param importance.")
        while self.status == State.RUNNING:
            import time
            from IPython.display import clear_output

            time.sleep(time_interval)
            if len(self.trials[~self.trials["value"].isnull()]) > ntrials:
                if plot_func == "_plot_param_importances":
                    if len(self.trials[~self.trials["value"].isnull()]) >= 4:
                        clear_output(wait=True)
                        getattr(plot, plot_func)(
                            study=self._study, fig_size=fig_size, **kwargs
                        )
                        clear_output(wait=True)
                else:
                    getattr(plot, plot_func)(
                        study=self._study, fig_size=fig_size, **kwargs
                    )
                    clear_output(wait=True)
            if len(self.trials) == 0:
                plt.figure()
                plt.title("Intermediate Values Plot")
                plt.xlabel("Step")
                plt.ylabel("Intermediate Value")
                plt.show(block=False)

            ntrials = len(self.trials[~self.trials["value"].isnull()])
        getattr(plot, plot_func)(study=self._study, fig_size=fig_size, **kwargs)

    def plot_best_scores(
        self,
        best=True,  # type: bool
        inferior=True,  # type: bool
        time_interval=1,  # type: float
        fig_size=(800, 500),  # type: tuple
    ):
        """Plot optimization history of all trials in a study.

        Parameters
        ----------
        best:
            controls whether to plot the lines for the best scores so far.
        inferior:
            controls whether to plot the dots for the actual objective scores.
        time_interval:
            how often(in seconds) the plot refresh to check on the new trial results.
        fig_size: tuple
            width and height of the figure.

        Returns
        -------
        None
            Nothing.
        """
        self._plot(
            "_optimization_history",
            "_get_optimization_history_plot",
            time_interval=time_interval,
            fig_size=fig_size,
            best=best,
            inferior=inferior,
        )

    @runtime_dependency(module="optuna", install_from=OptionalDependency.OPTUNA)
    def plot_param_importance(
        self,
        importance_evaluator="Fanova",  # type: str
        time_interval=1,  # type: float
        fig_size=(800, 500),  # type: tuple
    ):
        """Plot hyperparameter importances.

        Parameters
        ----------
        importance_evaluator: str
            Importance evaluator. Valid values: "Fanova", "MeanDecreaseImpurity". Defaults
            to "Fanova".
        time_interval: float
            How often the plot refresh to check on the new trial results.
        fig_size: tuple
            Width and height of the figure.

        Raises
        ------
        :class:`NotImplementedErorr`
            Raised for unsupported importance evaluators

        Returns
        -------
        None
            Nothing.
        """
        assert importance_evaluator in [
            "MeanDecreaseImpurity",
            "Fanova",
        ], "Only support <code>MeanDecreaseImpurity</code> and <code>Fanova</code>."
        if importance_evaluator == "Fanova":
            evaluator = None
        elif importance_evaluator == "MeanDecreaseImpurity":
            evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
        else:
            raise NotImplemented(
                f"{importance_evaluator} is not supported. It can be either `Fanova` or `MeanDecreaseImpurity`."
            )
        try:
            self._plot(
                plot_module="_param_importances",
                plot_func="_plot_param_importances",
                time_interval=time_interval,
                fig_size=fig_size,
                evaluator=evaluator,
            )
        except:
            logger.error(
                msg="""Cannot calculate the hyperparameter importance. Increase the number of trials or time budget. """
            )

    def plot_intermediate_scores(
        self,
        time_interval=1,  # type: float
        fig_size=(800, 500),  # type: tuple
    ):
        """
        Plot intermediate values of all trials in a study.

        Parameters
        ----------
        time_interval: float
            Time interval for the plot. Defaults to 1.
        fig_size: tuple[int, int]
            Figure size. Defaults to (800, 500).

        Returns
        -------
        None
            Nothing.
        """
        if not self._enable_pruning:
            logger.error(
                msg="Pruning was not used during tuning. "
                "There are no intermediate values to plot."
            )

        self._plot(
            "_intermediate_values",
            "_get_intermediate_plot",
            time_interval=time_interval,
            fig_size=fig_size,
        )

    def plot_edf_scores(
        self,
        time_interval=1,  # type: float
        fig_size=(800, 500),  # type: tuple
    ):
        """
        Plot the EDF (empirical distribution function) of the scores.

        Only completed trials are used.

        Parameters
        ----------
        time_interval: float
            Time interval for the plot. Defaults to 1.
        fig_size: tuple[int, int]
            Figure size. Defaults to (800, 500).

        Returns
        -------
        None
            Nothing.
        """
        self._plot(
            "_edf", "_get_edf_plot", time_interval=time_interval, fig_size=fig_size
        )

    def plot_contour_scores(
        self,
        params=None,  # type: Optional[List[str]]
        time_interval=1,  # type: float
        fig_size=(800, 500),  # type: tuple
    ):
        """
        Contour plot of the scores.

        Parameters
        ----------
        params: Optional[List[str]]
            Parameter list to visualize. Defaults to all.
        time_interval: float
            Time interval for the plot. Defaults to 1.
        fig_size: tuple[int, int]
            Figure size. Defaults to (800, 500).

        Returns
        -------
        None
            Nothing.
        """
        validate_params_for_plot(params, self._param_distributions)
        try:
            self._plot(
                "_contour",
                "_get_contour_plot",
                time_interval=time_interval,
                fig_size=fig_size,
                params=params,
            )
        except ValueError:
            logger.warning(
                msg="Cannot plot contour score."
                " Increase the number of trials or time budget."
            )

    def plot_parallel_coordinate_scores(
        self,
        params=None,  # type: Optional[List[str]]
        time_interval=1,  # type: float
        fig_size=(800, 500),  # type: tuple
    ):
        """
        Plot the high-dimentional parameter relationships in a study.

        Note that, If a parameter contains missing values, a trial with missing values is not plotted.

        Parameters
        ----------
        params: Optional[List[str]]
            Parameter list to visualize. Defaults to all.
        time_interval: float
            Time interval for the plot. Defaults to 1.
        fig_size: tuple[int, int]
            Figure size. Defaults to (800, 500).

        Returns
        -------
        None
            Nothing.
        """
        validate_params_for_plot(params, self._param_distributions)
        self._plot(
            "_parallel_coordinate",
            "_get_parallel_coordinate_plot",
            time_interval=time_interval,
            fig_size=fig_size,
            params=params,
        )
