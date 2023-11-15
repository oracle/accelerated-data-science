#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import subprocess
from io import StringIO
import tempfile
import os
import datetime
import shutil
import sys
import glob
import stat
import conda_pack

import yaml
import argparse


def main(pack_folder_path, manifest_file=None):
    slug = os.path.basename(pack_folder_path)
    manifest_path = (
        manifest_file or glob.glob(os.path.join(pack_folder_path, "*_manifest.yaml"))[0]
    )
    with open(manifest_path) as f:
        env = yaml.safe_load(f.read())
    
    with tempfile.TemporaryDirectory() as td:
        process = subprocess.Popen(
            ["conda", "env", "export", "--prefix", pack_folder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        
        if process.returncode and stderr:
            print(stderr)
            raise Exception(
                f"Error export environment information from {pack_folder_path}"
            )
        try:
            new_env_info = yaml.safe_load(StringIO(stdout.decode("utf-8")))
        except Exception as e:
            print(f"Error reading dependency list from {stdout}")
            raise e

        manifest = env["manifest"]
        manifest["type"] = "published"
        new_env_info["manifest"] = manifest
        with open(manifest_path, "w") as f:
            yaml.safe_dump(new_env_info, f)
        pack_file = os.path.join(td, f"{slug}.tar.gz")
        conda_pack.pack(
            prefix=pack_folder_path,
            compress_level=7,
            output=pack_file,
            n_threads=-1,
            ignore_missing_files=True,
        )
        if not os.path.exists(pack_file):
            raise RuntimeError(
                "Error creating the pack file using `conda_pack.pack()`."
            )
        print(f"Copy {pack_file} to {pack_folder_path}")
        shutil.copy(pack_file, pack_folder_path)
        file_path = os.path.join(pack_folder_path, os.path.basename(pack_file))
        print(f"Pack built at {file_path}")
        print(
            f"changing permission for {file_path}",
            flush=True,
        )
        os.chmod(file_path, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Prepare conda archive",
        description="Uses conda_pack library to pack the conda environment.",
    )
    parser.add_argument("--conda-path", type=str, help="Path to the conda environment")
    parser.add_argument(
        "--manifest-location", type=str, default=None, help="Path to manifest location"
    )
    args = parser.parse_args()

    main(args.conda_path, args.manifest_location)
