#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import MagicMock, patch

import pytest
from ads.common.error import ChangesNotCommitted
from ads.model.model_metadata import ModelProvenanceMetadata
from git import Repo
import git


class TestModelProvenanceMetadata:
    def setup_method(self):
        self.repo = Repo.init("./fake_folder", mkdir=True)
        self.git_branch = "master"
        self.git_commit = "99ad04c31803f1d4ffcc3bf4afbd6bcf69a06af2"
        self.repository_url = "file:///home/datascience"

    def test_init(self):
        git_repo_details = ModelProvenanceMetadata(
            self.repo,
            self.git_branch,
            self.git_commit,
        )
        git_repo_details.repo == self.repo
        git_repo_details.git_commit == self.git_commit
        git_repo_details.git_branch == self.git_branch
        git_repo_details.repository_url == self.repository_url

    def test_fetch_repo_details(self):
        with patch("git.Repo") as mock_repo:
            repo = MagicMock()
            remotes = MagicMock()
            remotes.__len__.return_value = 1
            remote = MagicMock()
            remote.url = "remote_url"
            remotes.values.return_value = [remote]
            repo.remotes = remotes
            repo.working_dir = os.path.abspath("./fake_folder")
            repo.is_dirty.return_value = False
            mock_repo.return_value = repo

            head = MagicMock()
            commit = MagicMock()
            commit.hexsha = self.git_commit
            head.commit = commit
            repo.head = head
            repo.active_branch = self.git_branch

            git_repo_details = ModelProvenanceMetadata.fetch_training_code_details()
            assert git_repo_details.repository_url == "remote_url"
            assert git_repo_details.git_commit == self.git_commit
            assert git_repo_details.git_branch == self.git_branch

    def test_fetch_repo_details_with_training_script_path(self):
        with patch("os.path.exists") as mock_path:
            mock_path.return_value = True

            # /test is an invalid folder so git.Repo will throw error when it's passed as git directory.
            with pytest.raises(git.exc.InvalidGitRepositoryError):
                ModelProvenanceMetadata.fetch_training_code_details(
                    training_script_path="/test/code.py"
                )

    def test_fetch_repo_details_without_git_commit(self):
        with patch("git.Repo") as mock_repo:
            repo = MagicMock()
            remotes = MagicMock()
            remotes.__len__.return_value = 1
            remote = MagicMock()
            remote.url = "remote_url"
            remotes.values.return_value = [remote]
            repo.remotes = remotes
            repo.working_dir = os.path.abspath("./fake_folder")
            repo.is_dirty.return_value = False
            mock_repo.return_value = repo

            head = MagicMock()
            commit = MagicMock()
            commit.hexsha = ""
            head.commit = commit
            repo.head = head
            repo.active_branch = self.git_branch

            git_repo_details = ModelProvenanceMetadata.fetch_training_code_details()
            assert git_repo_details.git_commit == None

    def test_assert_path_not_dirty(self):
        with patch("git.Repo") as mock_repo:
            repo = MagicMock()
            remotes = MagicMock()
            remotes.__len__.return_value = 1
            remote = MagicMock()
            remote.url = "remote_url"
            remotes.values.return_value = [remote]
            repo.remotes = remotes
            repo.working_dir = os.path.abspath("./fake_folder")
            repo.is_dirty.return_value = True
            mock_repo.return_value = repo
            head = MagicMock()
            commit = MagicMock()
            commit.hexsha = self.git_commit
            head.commit = commit
            repo.head = head
            repo.active_branch = self.git_branch

            git_repo_details = ModelProvenanceMetadata.fetch_training_code_details()
            with pytest.raises(ChangesNotCommitted):
                git_repo_details.assert_path_not_dirty("./fake_folder", ignore=False)
