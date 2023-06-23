#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
Tests notebooks in a directory by running all cells.
This module does NOT use pytest.
"""

import os
import re
import subprocess
import sys
import tarfile
import tempfile
import traceback
from collections import OrderedDict
from typing import List

import nbformat
import requests


def run_command(cmd):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    out, err = process.communicate()
    std_out = out.decode()
    std_err = err.decode()
    return std_out, std_err, process.returncode


def run_notebooks(notebook_dir: str, notebook_list: List[str]):
    """Execute a notebook via nbconvert and collect output."""
    notebook_dir = os.path.abspath(notebook_dir)
    exit_code = 0
    errors_dict = OrderedDict()
    files = [os.path.join(notebook_dir, f) for f in notebook_list]

    print("=" * 100)
    print(f"Testing {len(files)} notebooks in {notebook_dir}...")
    print("=" * 100)

    for notebook_path in files:
        notebook_path = os.path.abspath(notebook_path)
        notebook_name = os.path.basename(notebook_path)
        print(f"{notebook_name:<50}", end="", flush=True)
        std_out = ""
        std_err = ""

        # Test will be considered as failed if there is an exception
        # or non-zero return code or cell error.
        cell_errors = []
        return_code = -1

        try:
            with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
                args = [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=-1",
                    "--ExecutePreprocessor.kernel_name=python3",
                    "--output",
                    fout.name,
                    notebook_path,
                ]

                # subprocess.check_call(args)
                std_out, std_err, return_code = run_command(" ".join(args))
                # Check notebook cells only if return code is 0
                if return_code == 0:
                    fout.seek(0)
                    nb = nbformat.read(fout, nbformat.current_nbformat)
                    cell_errors = [
                        output
                        for cell in nb.cells
                        if "outputs" in cell
                        for output in cell["outputs"]
                        if output.output_type == "error"
                    ]
        except:
            std_err += "EXCEPTION:\n" + traceback.format_exc()
            return_code = -1

        if return_code != 0 or cell_errors:
            exit_code = 1
            print("-- FAILED --", flush=True)
            logs = f"CODE:{return_code}\nSTD OUT:\n{std_out}\nSTD ERR:{std_err}\n"
            if cell_errors:
                logs += "CELL ERR:" + "\n".join(cell_errors)
            errors_dict[notebook_path] = logs
        else:
            print("** PASSED **", flush=True)

    if errors_dict:
        print("=" * 100)
        for notebook_path, errors in errors_dict.items():
            print("-" * 100)
            print(notebook_path)
            print("-" * 100)
            print(errors)

        print("*" * 100)
        print(f"Notebooks failed: {len(errors_dict)}")
        print("*" * 100)

    return exit_code


def fetch_latest_nb_url(artifactory_url: str):
    """Gets the freshest notebook file name from the artifactory."""
    response = requests.get(f"{artifactory_url}/notebooks/all/")
    if response.status_code == 200:
        version = max(
            re.findall(
                r"\d+",
                ";".join(
                    set(
                        re.findall(
                            r"(notebooks_\d{10}.tar)", response.content.decode("utf-8")
                        )
                    )
                ),
            )
        )
        return f"{artifactory_url}/notebooks/all/notebooks_{version}.tar"
    else:
        print(
            f"No notebooks found at {artifactory_url}/notebooks/all. Status code: {response.status_code}"
        )
        return None


def download_notebooks_archive(notebook_dir: str, artifactory_url: str):
    """Downloads the notebooks archive from the artifactory."""
    notebook_dir = os.path.abspath(notebook_dir)
    os.makedirs(notebook_dir, exist_ok=True)
    notebooks_url = fetch_latest_nb_url(artifactory_url)
    print(f"Fetching notebooks archive from : {notebooks_url}")
    response = requests.get(notebooks_url)
    if response.status_code == 200:
        notebook_file = os.path.join(notebook_dir, "notebooks.tar")
        print(f"Saving notebooks archive to {notebook_file}")
        with open(notebook_file, "wb") as outfile:
            outfile.write(response.content)
        print("Extracting notebooks archive.")
        with tarfile.open(notebook_file, mode="r:gz") as tar:
            tar.extractall(notebook_dir)


if __name__ == "__main__":
    notebook_dir = sys.argv[1]
    artifactory_url = sys.argv[2]
    notebook_list = sys.argv[3].replace(" ", "").strip("[]").split(",")
    download_notebooks_archive(notebook_dir, artifactory_url)
    sys.exit(run_notebooks(notebook_dir, notebook_list))
