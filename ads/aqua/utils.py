#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA utils and constants."""
import base64
import logging
import random
import re
import sys
from string import Template
from typing import List

import fsspec

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ODSC_AQUA")

UNKNOWN = ""
README = "README.md"
DEPLOYMENT_CONFIG = "deployment_config.json"
UNKNOWN_JSON_STR = "{}"


def get_logger():
    return logger


def random_color_generator(word: str):
    seed = sum([ord(c) for c in word]) % 13
    random.seed(seed)
    r = random.randint(10, 245)
    g = random.randint(10, 245)
    b = random.randint(10, 245)

    text_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.5 else "white"

    return f"#{r:02x}{g:02x}{b:02x}", text_color


def svg_to_base64_datauri(svg_contents: str):
    base64_encoded_svg_contents = base64.b64encode(svg_contents.encode())
    return "data:image/svg+xml;base64," + base64_encoded_svg_contents.decode()


def create_word_icon(label: str, width: int = 150, return_as_datauri=True):
    match = re.findall(r"(^[a-zA-Z]{1}).*?(\d+[a-z]?)", label)
    icon_text = "".join(match[0] if match else [label[0]])
    icon_color, text_color = random_color_generator(label)
    cx = width / 2
    cy = width / 2
    r = width / 2
    fs = int(r / 25)

    t = Template(
        """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="${width}" height="${width}">

            <style>
                text {
                    font-size: ${fs}em;
                    font-family: lucida console, Fira Mono, monospace;
                    text-anchor: middle;
                    stroke-width: 1px;
                    font-weight: bold;
                    alignment-baseline: central;
                }

            </style>

            <circle cx="${cx}" cy="${cy}" r="${r}" fill="${icon_color}" />
            <text x="50%" y="50%" fill="${text_color}">${icon_text}</text>
        </svg>
    """.strip()
    )

    icon_svg = t.substitute(**locals())
    if return_as_datauri:
        return svg_to_base64_datauri(icon_svg)
    else:
        return icon_svg


def get_artifact_path(custom_metadata_list: List) -> str:
    """Get the artifact path from the custom metadata list of model.

    Parameters
    ----------
    custom_metadata_list: List
        A list of custom metadata of model.

    Returns
    -------
    str:
        The artifact path from model.
    """
    for custom_metadata in custom_metadata_list:
        if custom_metadata.key == "Object Storage Path":
            return custom_metadata.value
    logger.debug("Failed to get artifact path from custom metadata.")
    return UNKNOWN


def read_file(file_path: str, **kwargs) -> str:
    try:
        with fsspec.open(file_path, "r", **kwargs.get("auth", {})) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}. {e}")
        return UNKNOWN
