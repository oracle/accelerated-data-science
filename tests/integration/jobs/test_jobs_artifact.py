#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import unittest
from ads.jobs.builders.runtimes.artifact import Artifact
from tests.integration.config import secrets


class JobArtifactCopyTest(unittest.TestCase):
    def assert_copy_job_artifact(self, expected_files, from_uri, unpack=False):
        with tempfile.TemporaryDirectory() as working_dir:
            Artifact.copy_from_uri(from_uri, working_dir, unpack)
            file_paths = []
            for dir_path, _, filenames in os.walk(working_dir):
                dir_relative_path = os.path.relpath(dir_path, working_dir)
                file_paths += [
                    os.path.join(dir_relative_path, file) for file in filenames
                ]
            # file_path has the format of "./relative/path/to/file"
            file_paths = [
                file_path for file_path in file_paths if "__pycache__" not in file_path
            ]

            self.assertEqual(len(file_paths), len(expected_files), str(file_paths))
            for relative_path in expected_files:
                self.assertIn(relative_path, file_paths)

    def test_copy_file_from_local(self):
        self.assert_copy_job_artifact(
            ["./plot.ipynb"],
            os.path.join(os.path.dirname(__file__), "../fixtures/plot.ipynb"),
        )

    def test_copy_dir_from_local(self):
        self.assert_copy_job_artifact(
            [
                "job_archive/my_package/__init__.py",
                "job_archive/my_package/entrypoint_ads.py",
                "job_archive/my_package/entrypoint.py",
                "job_archive/my_package/utils.py",
                "job_archive/test_notebook.ipynb",
                "job_archive/main.py",
                "job_archive/my_module.py",
                "job_archive/script.sh",
            ],
            os.path.join(os.path.dirname(__file__), "../fixtures/job_archive"),
        )

    def test_copy_file_from_oci(self):
        self.assert_copy_job_artifact(
            ["./script.sh"],
            f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/job_artifact/script.sh",
        )

    def test_copy_dir_from_oci(self):
        self.assert_copy_job_artifact(
            [
                "job_artifact/script.sh",
                "job_artifact/job_archive.zip",
                "job_artifact/folder/script.sh",
            ],
            f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/job_artifact/",
        )

    def test_copy_file_from_https(self):
        self.assert_copy_job_artifact(
            ["./beginner.ipynb"],
            "https://github.com/tensorflow/docs/raw/master/site/en/tutorials/quickstart/beginner.ipynb",
            "beginner.ipynb",
        )

    def test_copy_zip_from_oci(self):
        self.assert_copy_job_artifact(
            [
                "job_archive/my_package/__init__.py",
                "job_archive/my_package/entrypoint_ads.py",
                "job_archive/my_package/entrypoint.py",
                "job_archive/my_package/utils.py",
                "job_archive/main.py",
                "job_archive/my_module.py",
                "job_archive/script.sh",
            ],
            f"oci://{secrets.jobs.BUCKET_B}@{secrets.common.NAMESPACE}/job_artifact/job_archive.zip",
            unpack=True,
        )
