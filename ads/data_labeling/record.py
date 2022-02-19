#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import asdict, dataclass
from typing import Any, Tuple, Union, Dict, List

from ads.data_labeling.boundingbox import BoundingBoxItem
from ads.data_labeling.ner import NERItem


@dataclass
class Record:
    """Class representing Record.

    Attributes
    ----------
    path: str
        File path.
    content: Any
        Content of the record.
    annotation: Union[Tuple, str, List[BoundingBoxItem], List[NERItem]]
        Annotation/label of the record.
    """

    path: str = ""
    content: Any = None
    annotation: Union[Tuple, str, List[BoundingBoxItem], List[NERItem]] = None

    def to_dict(self) -> Dict:
        """Convert the Record instance to a dictionary.

        Returns
        -------
        Dict
            Dictionary representation of the Record instance.
        """
        return asdict(self)

    def to_tuple(
        self,
    ) -> Tuple[str, Any, Union[Tuple, str, List[BoundingBoxItem], List[NERItem]]]:
        """Convert the Record instance to a tuple.

        Returns
        -------
        Tuple
            Tuple representation of the Record instance.
        """
        return (self.path, self.content, self.annotation)
