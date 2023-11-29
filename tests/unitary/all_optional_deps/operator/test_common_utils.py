#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import os
import unittest
from unittest.mock import patch, MagicMock

from ads.opctl.operator.common.utils import _build_image, _parse_input_args


class TestBuildImage(unittest.TestCase):
    def setUp(self):
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.dockerfile = os.path.join(self.curr_dir, "test_files", "Dockerfile.test")
        self.image_name = "test_image"
        self.tag = "test_tag"
        self.target = "test_target"
        self.kwargs = {"arg1": "value1", "arg2": "value2"}

    @patch("ads.opctl.utils.run_command")
    @patch("time.time")
    def test_build_image(self, mock_time, mock_run_command):
        mock_time.return_value = 1
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run_command.return_value = mock_proc

        image_name = _build_image(
            self.dockerfile,
            self.image_name,
            tag=self.tag,
            target=self.target,
            **self.kwargs,
        )

        expected_image_name = f"{self.image_name}:{self.tag}"
        command = [
            "docker",
            "build",
            "-t",
            expected_image_name,
            "-f",
            self.dockerfile,
            "--target",
            self.target,
            "--build-arg",
            "RND=1",
            os.path.dirname(self.dockerfile),
        ]

        mock_run_command.assert_called_once_with(command)
        self.assertEqual(image_name, expected_image_name)

    def test_build_image_missing_dockerfile(self):
        with self.assertRaises(FileNotFoundError):
            _build_image("non_existing_docker_file", "non_existing_image")

    def test_build_image_missing_image_name(self):
        with self.assertRaises(ValueError):
            _build_image(self.dockerfile, None)

    @patch("ads.opctl.utils.run_command")
    def test_build_image_docker_build_failure(self, mock_run_command):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_run_command.return_value = mock_proc

        with self.assertRaises(RuntimeError):
            _build_image(self.dockerfile, self.image_name)


class TestParseInputArgs(unittest.TestCase):
    def test_parse_input_args_with_file(self):
        raw_args = ["-f", "path/to/file.yaml"]
        expected_output = (
            argparse.Namespace(file="path/to/file.yaml", spec=None, verify=False),
            [],
        )
        self.assertEqual(_parse_input_args(raw_args), expected_output)

    def test_parse_input_args_with_spec(self):
        raw_args = ["-s", "spec"]
        expected_output = (argparse.Namespace(file=None, spec="spec", verify=False), [])
        self.assertEqual(_parse_input_args(raw_args), expected_output)

    def test_parse_input_args_with_verify(self):
        raw_args = ["-v", "True"]
        expected_output = (argparse.Namespace(file=None, spec=None, verify=True), [])
        self.assertEqual(_parse_input_args(raw_args), expected_output)

    def test_parse_input_args_with_unknown_args(self):
        raw_args = ["-f", "path/to/file.yaml", "--unknown-arg", "value"]
        expected_output = (
            argparse.Namespace(file="path/to/file.yaml", spec=None, verify=False),
            ["--unknown-arg", "value"],
        )
        self.assertEqual(_parse_input_args(raw_args), expected_output)

    @patch("argparse.ArgumentParser.parse_known_args")
    def test_parse_input_args_with_no_args(self, mock_parse_known_args):
        mock_parse_known_args.return_value = (
            argparse.Namespace(file=None, spec=None, verify=False),
            [],
        )
        expected_output = (argparse.Namespace(file=None, spec=None, verify=False), [])
        self.assertEqual(_parse_input_args([]), expected_output)
