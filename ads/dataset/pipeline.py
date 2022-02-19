#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import uuid

from sklearn.pipeline import Pipeline

from ads.dataset import helper


class TransformerPipeline(Pipeline):
    def __init__(self, steps):
        assert len(steps) > 0, "steps is empty"
        self.step_names = set()
        steps = [self._get_step(step) for step in steps]
        super(TransformerPipeline, self).__init__(steps=steps)

    def add(self, transformer):
        """
        Add transformer to data transformation pipeline

        Parameters
        ----------
        transformer: Union[TransformerMixin, tuple(str, TransformerMixin)]
               if tuple, (name, transformer implementing transform)

        """
        step = self._get_step(transformer)
        self.steps.append(step)

    def _get_step(self, step):
        """
        Generate unique step name and transformer to add in pipeline.

        Parameters
        ----------
        step: transformer instance implementing fit and transform methods

        Retuns
        ------
        (unique_step_name, transformer) : tuple
        """
        if isinstance(step, tuple):
            name = step[0]
            step = step[1]
        else:
            name = step.__class__.__name__
            step = step
        if name in self.step_names:
            name = name + "/" + str(uuid.uuid4())
        self.step_names.add(name)
        return name, step

    def visualize(self):
        helper.visualize_transformation(self)
