#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import os
import click


@click.command()
@click.argument("output_path")
@click.option(
    "--source-folder",
    "-s",
    help="source folder to search for environment yaml",
    required=False,
    default=None,
)
def merge(output_path, source_folder):
    merged = _merge(source_folder)
    with open(output_path, "w") as f:
        yaml.safe_dump(merged, f)


def _merge(source_folder=None):
    conda_dependencies = set([])
    pip_dependencies = set([])
    if not source_folder:
        source_folder = os.path.join("operators")
    for dirpath, dirnames, filenames in os.walk(source_folder):
        for fname in filenames:
            if fname == "environment.yaml":
                env_yaml = os.path.join(dirpath, fname)
                print(env_yaml)
                with open(env_yaml, "r") as f:
                    dependencies = yaml.safe_load(f.read())["dependencies"]
                for dep in dependencies:
                    if isinstance(dep, dict) and "pip" in dep:
                        pip_dependencies.update(dep["pip"])
                    else:
                        conda_dependencies.add(dep)
    conda_dependencies.add("pip")
    merged_dependencies = {
        "dependencies": list(conda_dependencies) + [{"pip": list(pip_dependencies)}]
    }
    return merged_dependencies


if __name__ == "__main__":
    merge()
