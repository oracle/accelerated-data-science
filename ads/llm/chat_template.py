#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os


class ChatTemplates:
    """Contains chat templates."""

    @staticmethod
    def _read_template(filename):
        with open(
            os.path.join(os.path.dirname(__file__), "templates", filename),
            mode="r",
            encoding="utf-8",
        ) as f:
            return f.read()

    @staticmethod
    def mistral():
        """Chat template for auto tool calling with Mistral model deploy with vLLM."""
        return ChatTemplates._read_template("tool_chat_template_mistral_parallel.jinja")

    @staticmethod
    def hermes():
        """Chat template for auto tool calling with Hermes model deploy with vLLM."""
        return ChatTemplates._read_template("tool_chat_template_hermes.jinja")
