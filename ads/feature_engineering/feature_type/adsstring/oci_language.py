#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import Dict, List

import oci.ai_language
import pandas as pd
from ads.common import auth as authutil
from ads.common import oci_client


class OCILanguage(object):  # pragma: no cover
    """Defines the OCILanguage plugin for ADSString built on top of the OCI Language Services.

    Example
    -------
    >>> ADSString.plugin_register(OCILanguage)
    >>> s = ADSString("This movie is awesome.")
    >>> s.absa
    >>> s.text_classification
    >>> s.ner
    >>> s.language_dominant
    """

    def __init__(self, auth=None):
        auth = auth if auth else authutil.default_signer()
        self.ai_client = oci_client.OCIClientFactory(**auth).ai_language

    @property
    def ner(self) -> List[Dict]:
        """Detects named entites in the text."""
        output = self.ai_client.batch_detect_language_entities(
            oci.ai_language.models.BatchDetectLanguageEntitiesDetails(
                documents=[
                    oci.ai_language.models.TextDocument(key="1", text=self.string)
                ]
            )
        )
        return json.loads(str(output.data.documents[0]))["entities"]

    @property
    def language_dominant(self) -> List[Dict]:
        """Determines the language of the text along with ISO 639-1 language code and a probability score."""
        output = self.ai_client.batch_detect_dominant_language(
            oci.ai_language.models.BatchDetectDominantLanguageDetails(
                documents=[
                    oci.ai_language.models.DominantLanguageDocument(
                        key="1", text=self.string
                    )
                ]
            )
        )
        return json.loads(str(output.data.documents[0]))

    @property
    def key_phrase(self) -> List[Dict]:
        """Extracts the most relevant words from the ADSString object and assigns them a score."""
        output = self.ai_client.batch_detect_language_key_phrases(
            oci.ai_language.models.BatchDetectLanguageKeyPhrasesDetails(
                documents=[
                    oci.ai_language.models.TextDocument(key="1", text=self.string)
                ]
            )
        )
        return json.loads(str(output.data.documents[0]))["key_phrases"]

    @property
    def absa(self) -> List[Dict]:
        """Runs aspect-based sentiment analysis on the text to gauge teh mood or the tone of the text."""
        output = self.ai_client.batch_detect_language_sentiments(
            oci.ai_language.models.BatchDetectLanguageSentimentsDetails(
                documents=[
                    oci.ai_language.models.TextDocument(key="1", text=self.string)
                ]
            )
        )
        return json.loads(str(output.data.documents[0]))["aspects"]

    @property
    def text_classification(self) -> List[Dict]:
        """Analyses the text and identifies categories for the content with a confidence score."""
        output = self.ai_client.batch_detect_language_text_classification(
            oci.ai_language.models.BatchDetectLanguageTextClassificationDetails(
                documents=[
                    oci.ai_language.models.TextDocument(key="1", text=self.string)
                ]
            )
        )
        return json.loads(str(output.data.documents[0]))["text_classification"]
