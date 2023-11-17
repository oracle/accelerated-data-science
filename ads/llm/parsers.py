#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Any
from langchain.schema.output_parser import BaseOutputParser


class SingleKeyDictOutputParser(BaseOutputParser[Any]):
    """Stores the text into a dictionary with a single key.

    Example::
        
        # Putting the output of the LLM into a dictionary
        chain = GenerativeAI(...) | SingleKeyDictOutputParser(key="my_key")
        # The ``output`` will be a dictionary like ``{"my_key": "..."}``
        output = chain.invoke("...")

    """

    key: str
    """The key in the dictionary. The the input text will be store as the value of this key."""

    def parse(self, text: str) -> dict:
        return {self.key: text}
