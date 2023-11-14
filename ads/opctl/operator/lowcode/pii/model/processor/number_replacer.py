#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import random
import re

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    import scrubadub
except ImportError:
    raise ModuleNotFoundError(
        f"`scrubadub` module was not found. Please run "
        f"`pip install {OptionalDependency.PII}`."
    )


class NumberReplacer(scrubadub.post_processors.PostProcessor):
    name = "number_replacer"
    _ENTITIES = [
        "number",
        "mrn",
        "fin",
        "phone",
        "social_security_number",
    ]

    @staticmethod
    def replace_digit(obj):
        return random.choice("0123456789")

    def match_entity_type(self, filth_types):
        if list(set(self._ENTITIES) & set(filth_types)):
            return True
        return False

    def replace_date(self, text):
        date_formats = ["%m-%d-%Y", "%m-%d-%y", "%d-%m-%Y", "%d-%m-%y"]
        for date_format in date_formats:
            try:
                date = datetime.datetime.strptime(text, date_format)
            except ValueError:
                continue
            if date.year < 1900 or date.year > datetime.datetime.now().year:
                continue
            # Now the date is a valid data between 1900 and now
            return text
        return None

    def replace(self, text):
        # Check dates
        date = self.replace_date(text)
        if date:
            return date
        return re.sub(r"\d", self.replace_digit, text)

    def process_filth(self, filth_list):
        for filth in filth_list:
            # Do not process it if it already has a replacement.
            if filth.replacement_string:
                continue
            if filth.type.lower() in self._ENTITIES:
                filth.replacement_string = self.replace(filth.text)
            # Replace the numbers for merged filth
            if filth.type.lower() == "unknown" and hasattr(filth, "filths"):
                filth_types = set([f.type for f in filth.filths])
                if self.match_entity_type(filth_types):
                    filth.replacement_string = self.replace(filth.text)
        return filth_list
