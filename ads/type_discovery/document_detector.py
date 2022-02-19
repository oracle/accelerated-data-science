#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import, division

import re

import pandas as pd

from ads.type_discovery import logger
from ads.type_discovery.abstract_detector import AbstractTypeDiscoveryDetector
from ads.type_discovery.typed_feature import DocumentTypedFeature, AddressTypedFeature


class DocumentDetector(AbstractTypeDiscoveryDetector):

    _min_cjk_chars_for_document = 100
    _min_words = 10
    _min_html_tags = 5

    _html_pattern = re.compile("<.*?>")

    _unicode_ranges = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
        {
            "from": ord(u"\U0002F800"),
            "to": ord(u"\U0002fa1f"),
        },  # compatibility ideographs
        {"from": ord(u"\u3040"), "to": ord(u"\u309f")},  # Japanese Hiragana
        {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {
            "from": ord(u"\U0002b820"),
            "to": ord(u"\U0002ceaf"),
        },  # included as of Unicode 8.0
    ]

    def _is_cjk_char(self, char):
        return any(
            [
                range["from"] <= ord(char) <= range["to"]
                for range in DocumentDetector._unicode_ranges
            ]
        )

    def cjk_string(self, document):
        cjk_char_count = sum([1 if self._is_cjk_char(c) else 0 for c in document])
        return cjk_char_count / len(document) >= 0.2

    def html_document(self, document):
        return (
            len(list(re.finditer(DocumentDetector._html_pattern, document)))
            > DocumentDetector._min_html_tags
        )

    def discover(self, name, series):
        #
        # very basic detection of a document. If the document is CJK then we use only the document length
        # otherwise we split on whitespace and confirm that there are word-like strings
        #
        if series.dtype == "object":
            null_series = series.loc[~series.isnull()]
            first_non_null_document = null_series.iloc[0]

            if isinstance(first_non_null_document, str):

                is_html = self.html_document(first_non_null_document)

                if self.cjk_string(first_non_null_document):
                    if (
                        len(first_non_null_document)
                        >= DocumentDetector._min_cjk_chars_for_document
                    ):
                        tf = DocumentTypedFeature.build(
                            name, series, is_cjk=True, is_html=is_html
                        )
                        logger.debug(
                            "type discovery on CJK column [{}]/[{}] found to be a document".format(
                                name, series.dtype
                            )
                        )
                        return tf
                else:
                    # find rows with above average length
                    above_avg_series = null_series.loc[
                        null_series.str.len() >= null_series.str.len().mean()
                    ]

                    # take a sample. max 500 docs
                    above_avg_series_sample = above_avg_series.sample(
                        n=min(500, len(above_avg_series))
                    )

                    # if all of the samples have more than min_words tokens..
                    mean_number_of_words = (
                        above_avg_series_sample.str.split().str.len().mean()
                    )
                    if mean_number_of_words > DocumentDetector._min_words:
                        if (
                            mean_number_of_words < 15
                            and above_avg_series_sample.str.count(",").mean()
                            / mean_number_of_words
                            > 0.1
                        ):
                            # many commas probably means address type
                            logger.debug(
                                "type discovery on column [{}]/[{}] looks like an address type".format(
                                    name, series.dtype
                                )
                            )
                            return AddressTypedFeature.build(name, series)
                        else:
                            logger.debug(
                                "type discovery on non-CJK column [{}]/[{}] found to be a document".format(
                                    name, series.dtype
                                )
                            )

                            # previous check of first document for HTML is now refined using longer documents
                            is_html = all(
                                [
                                    self.html_document(doc)
                                    for doc in above_avg_series_sample
                                ]
                            )

                            return DocumentTypedFeature.build(
                                name, series, is_cjk=False, is_html=is_html
                            )

        return False


if __name__ == "__main__":
    dd = DocumentDetector()
