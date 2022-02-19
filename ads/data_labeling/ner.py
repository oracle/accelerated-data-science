#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import List


class WrongEntityFormatLabelNotString(ValueError):
    def __init__(self):
        super().__init__("Wrong entity format. Label is not a String.")


class WrongEntityFormatOffsetNotInteger(ValueError):
    def __init__(self):
        super().__init__("Wrong entity format. Offset is not an Integer.")


class WrongEntityFormatLengthNotInteger(ValueError):
    def __init__(self):
        super().__init__("Wrong entity format. Length is not an Integer.")


class WrongEntityFormatOffsetIsNegative(ValueError):
    def __init__(self):
        super().__init__("Wrong entity format. Offset is not a Nonnegative.")


class WrongEntityFormatLengthIsNegative(ValueError):
    def __init__(self):
        super().__init__("Wrong entity format. Length is not a Nonnegative.")


class WrongEntityFormatLabelIsEmpty(ValueError):
    def __init__(self):
        super().__init__("Wrong entity format. Label is empty.")


@dataclass
class NERItem:
    """NERItem class which is a representation of a token span.

    Attributes
    ----------
    label: str
        Entity name.
    offset: int
        The token span's entity start index position in the text.
    length: int
        Length of the token span.
    """

    label: str = ""
    offset: int = 0
    length: int = 0

    def _validate(self):
        """Validates the instance.

        Raises
        ------
        WrongEntityFormat
           If the entity has a wrong format.

        """
        if not isinstance(self.label, str):
            raise WrongEntityFormatLabelNotString()
        if not (
            isinstance(self.offset, int)
            or (isinstance(self.offset, float) and self.offset.is_integer())
        ):
            raise WrongEntityFormatOffsetNotInteger()
        if not (
            isinstance(self.length, int)
            or (isinstance(self.length, float) and self.length.is_integer())
        ):
            raise WrongEntityFormatLengthNotInteger()
        if self.offset < 0:
            raise WrongEntityFormatOffsetIsNegative()
        if self.length < 0:
            raise WrongEntityFormatLengthIsNegative()
        if self.label == "":
            raise WrongEntityFormatLabelIsEmpty()

    def __post_init__(self):
        self._validate()

    def to_spacy(self) -> tuple:
        """Converts one NERItem to the spacy format.

        Returns
        -------
        Tuple
            NERItem in the spacy format
        """
        return (self.offset, self.offset + self.length, self.label)

    @classmethod
    def from_spacy(cls, token) -> "NERItem":
        offset, end, label = token
        return cls(label, offset, end - offset)


@dataclass
class NERItems:
    """NERItems class consists of a list of NERItem.

    Attributes
    ----------
    items: List[NERItem]
        List of NERItem.
    """

    items: List[NERItem] = field(default_factory=list)

    def __getitem__(self, index: int) -> NERItem:
        return self.items[index]

    def to_spacy(self) -> List[tuple]:
        """Converts NERItems to the spacy format.

        Returns
        -------
        List[tuple]
            List of NERItems in the Spacy format.
        """
        return [item.to_spacy() for item in self.items]
