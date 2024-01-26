#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os


def default_signer(**kwargs):
    os.environ["EXTRA_USER_AGENT_INFO"] = "Pii-Operator"
    from ads.common.auth import default_signer

    return default_signer(**kwargs)


def get_output_name(given_name, target_name=None):
    """Add ``-out`` suffix to the src filename."""
    if not target_name:
        basename = os.path.basename(given_name)
        fn, ext = os.path.splitext(basename)
        target_name = fn + "_out" + ext
    return target_name


def construct_filth_cls_name(name: str) -> str:
    """Constructs the filth class name from the given name.
    For example, "name" -> "NameFilth".

    Args:
        name (str): filth class name.

    Returns:
        str: The filth class name.
    """
    return "".join([s.capitalize() for s in name.split("_")]) + "Filth"


################
# Report utils #
################
def compute_rate(elapsed_time, num_unit):
    return elapsed_time / num_unit
