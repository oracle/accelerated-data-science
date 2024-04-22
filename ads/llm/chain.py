#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import logging
import os
import pathlib
from typing import Any, List, Optional

import yaml
from langchain.llms.base import LLM
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
    RunnableSequence,
)
from ads.llm.guardrails.base import GuardrailIO, Guardrail, RunInfo, BlockedByGuardrail


logger = logging.getLogger(__name__)
SPEC_CHAIN_TYPE = "_type"
SPEC_CHAIN = "chain"
LOG_ADS_GUARDRAIL_INFO = "LOG_ADS_GUARDRAIL_INFO"


class GuardrailSequence(RunnableSequence):
    """Represents a sequence of guardrails and other LangChain (non-guardrail) components."""

    first: Optional[Runnable] = None
    last: Optional[Runnable] = None

    raise_exception: bool = False
    """The ``raise_exception`` property indicate whether an exception should be raised
    if the content is blocked by one of the guardrails.
    This property is set to ``False`` by default.
    Note that each guardrail also has its own ``raise_exception`` property.
    This property on GuardrailSequence has no effect
    when the ``raise_exception`` is set to False on the individual guardrail.

    When this is ``False``, instead of raising an exception,
    the custom message from the guardrail will be returned as the output.

    When this is ``True``, the ``BlockedByGuardrail`` exception from the guardrail will be raised.
    """

    log_info: bool = False
    """Indicate whether to print the run info at the end of each invocation.
    This option can also be turned on if the environment variable LOG_ADS_GUARDRAIL_INFO is set to "1".
    """

    max_retry: int = 1
    """Maximum number of retry for running the Guardrail sequence again if the output is blocked by a guardrail."""

    @property
    def steps(self) -> List[Runnable[Any, Any]]:
        """Steps in the sequence."""
        if self.first:
            chain = [self.first] + self.middle
        else:
            return []
        if self.last:
            chain += [self.last]
        return chain

    @staticmethod
    def type() -> str:
        """A unique identifier as type for serialization."""
        return "ads_guardrail_sequence"

    @classmethod
    def from_sequence(cls, sequence: RunnableSequence, **kwargs):
        """Creates a GuardrailSequence from a LangChain runnable sequence."""
        return cls(
            first=sequence.first, middle=sequence.middle, last=sequence.last, **kwargs
        )

    def __or__(self, other) -> "GuardrailSequence":
        """Adds another component to the end of this sequence.
        If the sequence is empty, the component will be added as the first step of the sequence.
        """
        if not self.first:
            return GuardrailSequence(first=other)
        if not self.last:
            return GuardrailSequence(first=self.first, last=other)
        return self.from_sequence(super().__or__(other))

    def __ror__(self, other) -> "GuardrailSequence":
        """Chain this sequence to the end of another component."""
        return self.from_sequence(super().__ror__(other))

    def invoke(self, input: Any, config: RunnableConfig = None) -> GuardrailIO:
        """Invokes the guardrail.

        In LangChain interface, invoke() is designed for calling the chain with a single input,
        while batch() is designed for calling the chain with a list of inputs.
        https://python.langchain.com/docs/expression_language/interface

        """
        return self.run(input)

    def _invoke_llm(self, llm: LLM, texts: list, num_generations: int, **kwargs):
        if num_generations > 1:
            if len(texts) > 1:
                raise NotImplementedError(
                    "Batch completion with more than 1 prompt is not supported."
                )
            # TODO: invoke in parallel
            # TODO: let llm generate n completions.
            output = [llm.invoke(texts[0], **kwargs) for _ in range(num_generations)]
        else:
            output = llm.batch(texts, **kwargs)
        return output

    def _run_step(
        self, step: Runnable, obj: GuardrailIO, num_generations: int, **kwargs
    ):
        if not isinstance(step, Guardrail):
            # Invoke the step as a LangChain component
            spec = {}
            with RunInfo(name=step.__class__.__name__, input=obj.data) as info:
                if isinstance(step, LLM):
                    output = self._invoke_llm(step, obj.data, num_generations, **kwargs)
                    spec.update(kwargs)
                    spec["num_generations"] = num_generations
                else:
                    output = step.batch(obj.data)
                info.output = output
                info.parameters = {
                    "class": step.__class__.__name__,
                    "path": step.__module__,
                    "spec": spec,
                }
            obj.info.append(info)
            obj.data = output
        else:
            obj = step.invoke(obj)
        return obj

    def run(self, input: Any, num_generations: int = 1, **kwargs) -> GuardrailIO:
        """Runs the guardrail sequence.

        Parameters
        ----------
        input : Any
            Input for the guardrail sequence.
            This will be the input for the first step in the sequence.
        num_generations : int, optional
            The number of completions to be generated by the LLM, by default 1.

        The kwargs will be passed to LLM step(s) in the guardrail sequence.

        Returns
        -------
        GuardrailIO
            Contains the outputs and metrics from each step.
            The final output is stored in GuardrailIO.data property.
        """
        retry_count = 0
        while True:
            retry_count += 1
            obj = GuardrailIO(data=[input])
            try:
                for i, step in enumerate(self.steps):
                    obj = self._run_step(step, obj, num_generations, **kwargs)
                break
            except BlockedByGuardrail as ex:
                if retry_count < self.max_retry:
                    continue
                if self.raise_exception:
                    raise ex
                obj.data = [ex.message]
                obj.info.append(ex.info)
                break
        if self.log_info or os.environ.get(LOG_ADS_GUARDRAIL_INFO) == "1":
            # LOG_ADS_GUARDRAIL_INFO is set to "1" in score.py by default.
            print(obj.dict())
        # If the output is a singleton list, take it out of the list.
        if isinstance(obj.data, list) and len(obj.data) == 1:
            obj.data = obj.data[0]
        return obj

    def _save_to_file(self, chain_dict, filename, overwrite=False):
        expanded_path = os.path.expanduser(filename)
        if os.path.isfile(expanded_path) and not overwrite:
            raise FileExistsError(
                f"File {expanded_path} already exists."
                "Set overwrite to True if you would like to overwrite the file."
            )

        file_ext = pathlib.Path(expanded_path).suffix.lower()
        if file_ext not in [".yaml", ".json"]:
            raise ValueError(
                f"{self.__class__.__name__} can only be saved as yaml or json format."
            )
        with open(expanded_path, "w", encoding="utf-8") as f:
            if file_ext == ".yaml":
                yaml.safe_dump(chain_dict, f, default_flow_style=False)
            elif file_ext == ".json":
                json.dump(chain_dict, f)

    def save(self, filename: str = None, overwrite: bool = False) -> dict:
        """Serialize the sequence to a dictionary.
        Optionally, save the sequence into a JSON or YAML file.

        The dictionary will look like the following::

            {
                "_type": "ads_guardrail_sequence",
                "chain": [
                    ...
                ]
            }

        where ``chain`` contains a list of steps.

        Parameters
        ----------
        filename : str
            YAML or JSON filename to store the serialized sequence.

        Returns
        -------
        dict
            The sequence saved as a dictionary.
        """
        from ads.llm.serialize import dump

        chain_spec = []
        for step in self.steps:
            chain_spec.append(dump(step))
        chain_dict = {
            SPEC_CHAIN_TYPE: self.type(),
            SPEC_CHAIN: chain_spec,
        }

        if filename:
            self._save_to_file(chain_dict, filename, overwrite)

        return chain_dict

    @classmethod
    def load(cls, chain_dict: dict, **kwargs) -> "GuardrailSequence":
        """Loads the sequence from a dictionary config.

        Parameters
        ----------
        chain_dict : dict
            A dictionary containing the key "chain".
            The value of "chain" should be a list of dictionary.
            Each dictionary corresponds to a step in the chain.

        Returns
        -------
        GuardrailSequence
            A GuardrailSequence loaded from the config.
        """
        from ads.llm.serialize import load

        chain_spec = chain_dict[SPEC_CHAIN]
        steps = [load(config, **kwargs) for config in chain_spec]
        return cls(*steps)

    def __str__(self) -> str:
        return "\n".join([str(step.__class__) for step in self.steps])
