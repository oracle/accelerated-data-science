#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
from unittest.mock import patch

from ads.jobs.builders.runtimes.artifact import Artifact


class TestArtifact:
    """Contains tests for Artifact class and its methods in ads.jobs.builders.runtimes.artifact module."""

    @patch.object(Artifact, "_download_from_web")
    @patch("shutil.unpack_archive")
    def test_copy_from_uri(self, mock_unpack_archive, mock_download_from_web):
        """Tests unpacking archive file a directory as job artifact."""

        with tempfile.TemporaryDirectory() as working_dir:
            # uri IS NOT an archive file, so unpacking should NOT be called
            Artifact.copy_from_uri(
                uri="http://<path>/filezip", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_not_called()

            Artifact.copy_from_uri(
                uri="http://<path>/filetar.gz", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_not_called()

            Artifact.copy_from_uri(
                uri="http://<path>/filetar", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_not_called()

            Artifact.copy_from_uri(
                uri="http://<path>/filetgz", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_not_called()

            # uri IS an archive file, so unpacking should be called
            Artifact.copy_from_uri(
                uri="http://<path>/file.zip", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_called()

            Artifact.copy_from_uri(
                uri="http://<path>/file.tar.gz", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_called()

            Artifact.copy_from_uri(
                uri="http://<path>/file.tar", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_called()

            Artifact.copy_from_uri(
                uri="http://<path>/file.tgz", to_path=working_dir, unpack=True
            )
            mock_unpack_archive.assert_called()
