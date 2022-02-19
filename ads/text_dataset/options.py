#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from enum import Enum, auto
from typing import Any, List, Dict

from fsspec.core import OpenFile


class Options(Enum):
    FILE_NAME = auto()
    FILE_METADATA = auto()


class OptionHandler:
    def __init__(self, dataloader: "ads.text_dataset.dataset.DataLoader") -> None:
        self.dataloader = dataloader

    def handle(self, fhandler: OpenFile, spec: Any) -> Any:
        raise NotImplementedError()


class FileOption(OptionHandler):
    def handle(self, fhandler: OpenFile, spec: Any) -> Any:
        return fhandler.path


class MetadataOption(OptionHandler):
    def handle(self, fhandler: OpenFile, spec: Dict) -> List:
        metadata = self.dataloader.processor.get_metadata(fhandler)
        return [metadata.get(k, None) for k in spec["extract"]]


class OptionFactory:

    option_handlers = {
        Options.FILE_NAME: FileOption,
        Options.FILE_METADATA: MetadataOption,
    }

    @staticmethod
    def option_handler(option: Options) -> OptionHandler:
        handler = OptionFactory.option_handlers.get(option, None)
        if handler is None:
            raise RuntimeError(f"Option {option} Not Recognized.")
        return handler

    @classmethod
    def register_option(cls, option: Options, handler) -> None:
        cls.option_handlers[option] = handler
