#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


"""Contains post processors for scrubadub
Usage:

scrubber.add_post_processor(NameReplacer())
scrubber.add_post_processor(NumberReplacer())

To keep the same name replacement mappings across multiple documents,
either use the same scrubber instance to clean all the documents,
or use the same NameReplace() instance for all scrubbers.
"""
import datetime
import random
import re
import string
from typing import Sequence

import scrubadub
import gender_guesser.detector as gender_detector

from faker import Faker
from scrubadub.filth import Filth
from nameparser import HumanName


class NameReplacer(scrubadub.post_processors.PostProcessor):
    name = "name_replacer"

    def __init__(self, name: str = None, mapping: dict = None):
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = {}

        self.gender_detector = gender_detector.Detector()
        self.fake = Faker()
        self.groups = {
            "first": self.first_name_generator,
            "middle": self.first_name_generator,
            "last": self.last_name_generator,
            "suffix": lambda x: "",
        }
        super().__init__(name)

    def first_name_generator(self, name):
        detected_gender = self.gender_detector.get_gender(name)
        if "female" in detected_gender:
            return self.fake.first_name_female()
        elif "male" in detected_gender:
            return self.fake.first_name_male()
        return self.fake.first_name_nonbinary()

    def last_name_generator(self, *args):
        return self.fake.last_name()

    def unwrap_filth(self, filth_list):
        """Un-merge the filths if they have different types."""
        processed = []
        for filth in filth_list:
            # MergedFilths has the property "filths"
            # Do nothing if filth has a type already
            if filth.type in ["unknown", "", None] and hasattr(filth, "filths"):
                filth_types = set([f.type.lower() for f in filth.filths])
                # Do nothing if the filth does not contain a name
                if "name" not in filth_types:
                    processed.append(filth)
                    continue
                if len(filth_types) > 1:
                    processed.extend(filth.filths)
                    continue
                filth.type = filth.filths[0].type
                filth.detector_name = filth.filths[0].detector_name
            processed.append(filth)
        return processed

    @staticmethod
    def has_initial(name: HumanName) -> bool:
        for attr in ["first", "middle", "last"]:
            if len(str(getattr(name, attr)).strip(".")) == 1:
                return True
        return False

    @staticmethod
    def has_non_initial(name: HumanName) -> bool:
        for attr in ["first", "middle", "last"]:
            if len(str(getattr(name, attr)).strip(".")) > 1:
                return True
        return False

    @staticmethod
    def generate_component(name_component: str, generator):
        fake_component = generator(name_component)
        if len(name_component.rstrip(".")) == 1:
            fake_component = fake_component[0]
            if name_component.endswith("."):
                fake_component += "."
        return fake_component

    def save_name_mapping(self, name: HumanName, fake_name: HumanName):
        """Saves the names with initials to the mapping so that a new name will not be generated.
        For example, if name is "John Richard Doe", this method will save the following keys to the mapping:
        - J Doe
        - John D
        - J R Doe
        - John R D
        - John R Doe
        """
        # Both first name and last name must be presented
        if not name.first or not name.last:
            return
        # Remove any dot at the end of the name component.
        for attr in ["first", "middle", "last"]:
            setattr(name, attr, getattr(name, attr).rstrip("."))

        self.mapping[
            f"{name.first[0]} {name.last}"
        ] = f"{fake_name.first[0]} {fake_name.last}"

        self.mapping[
            f"{name.first} {name.last[0]}"
        ] = f"{fake_name.first} {fake_name.last[0]}"

        if name.middle:
            self.mapping[
                f"{name.first[0]} {name.middle[0]} {name.last}"
            ] = f"{fake_name.first[0]} {fake_name.middle[0]} {fake_name.last}"

            self.mapping[
                f"{name.first} {name.middle[0]} {name.last[0]}"
            ] = f"{fake_name.first} {fake_name.middle[0]} {fake_name.last[0]}"

            self.mapping[
                f"{name.first} {name.middle[0]} {name.last}"
            ] = f"{fake_name.first} {fake_name.middle[0]} {fake_name.last}"

    def replace(self, text):
        """Replaces a name with fake name.

        Parameters
        ----------
        text : str or HumanName
            The name to be replaced.
            If text is a HumanName object, the object will be modified to have the new fake names.

        Returns
        -------
        str
            The replaced name as text.
        """
        if isinstance(text, HumanName):
            name = text
        else:
            name = HumanName(text)
        skip = []
        # Check if the name is given with initial for one of the first name/last name
        key = None
        if self.has_initial(name) and self.has_non_initial(name):
            if name.middle:
                key = f'{name.first.rstrip(".")} {name.middle.rstrip(".")} {name.last.rstrip(".")}'
            else:
                key = f'{name.first.rstrip(".")} {name.last.rstrip(".")}'
            fake_name = self.mapping.get(key)
            # If a fake name is found matching the first initial + last name or first name + last initial
            # Replace the the initial with the corresponding initial
            # and skip processing the first and last name in the replacement.
            if fake_name:
                fake_name = HumanName(fake_name)
                name.first = fake_name.first
                name.last = fake_name.last
                skip = ["first", "last"]
                if name.middle:
                    name.middle = fake_name.middle
                    skip.append("middle")
        # Replace each component in the name
        for attr, generator in self.groups.items():
            if attr in skip:
                continue
            name_component = getattr(name, attr, None)
            if not name_component:
                continue
            # Check if a fake name has been generated for this name
            fake_component = self.mapping.get(name_component)
            if not fake_component:
                fake_component = self.generate_component(name_component, generator)
                # Generate a unique fake name that is not already in the mapping
                while fake_component and (
                    fake_component in self.mapping.keys()
                    or fake_component in self.mapping.values()
                ):
                    fake_component = self.generate_component(name_component, generator)
                self.mapping[name_component] = fake_component
            setattr(name, attr, fake_component)

        # Save name with initials to mapping
        original_name = text if isinstance(text, HumanName) else HumanName(text)
        self.save_name_mapping(original_name, name)
        return str(name)

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
        filth_list = self.unwrap_filth(filth_list)

        name_filths = []
        # Filter to keep only the names
        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "name":
                continue
            name_filths.append(filth)

        # Sort reverse by last name so that names having a last name will be processed first.
        # When a name is referred by last name (e.g. Mr. White), HumanName will parse it as first name.
        name_filths.sort(key=lambda x: HumanName(x.text).last, reverse=True)
        for filth in name_filths:
            filth.replacement_string = self.replace(filth.text)
        return filth_list


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

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
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


class EmailReplacer(scrubadub.post_processors.PostProcessor):
    name = "email_replacer"

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "email":
                continue
            filth.replacement_string = Faker().email()
        return filth_list


class HIBNReplacer(scrubadub.post_processors.PostProcessor):
    name = "hibn_replacer"

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
        # TODO: Add support for anomymizing Health insurance beneficiary number ~ Consecutive sequence of alphanumeric characters
        pass


class MBIReplacer(scrubadub.post_processors.PostProcessor):
    name = "mbi_replacer"
    CHAR_POOL = "ACDEFGHJKMNPQRTUVWXY"

    def generate_mbi(self):
        return "".join(random.choices(self.CHAR_POOL + string.digits, k=11))

    def process_filth(self, filth_list: Sequence[Filth]) -> Sequence[Filth]:
        for filth in filth_list:
            if filth.replacement_string:
                continue
            if filth.type.lower() != "mbi":
                continue
            filth.replacement_string = self.generate_mbi()
        return filth_list


POSTPROCESSOR_MAP = {
    item.name.lower(): item
    for item in [
        NameReplacer,
        NumberReplacer,
        EmailReplacer,
        HIBNReplacer,
        MBIReplacer,
    ]
}
