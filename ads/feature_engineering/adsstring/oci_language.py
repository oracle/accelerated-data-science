#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import oci.ai_language
import pandas as pd

from ads.common import auth as authutil
from ads.common import oci_client


class OCILanguage(object):  # pragma: no cover
    def __init__(self, auth=None):
        auth = auth if auth else authutil.default_signer()
        self.ai_client = oci_client.OCIClientFactory(**auth).ai_language

    @property
    def ner(self) -> pd.DataFrame:
        detect_language_entities_details = (
            oci.ai_language.models.DetectLanguageEntitiesDetails(text=self.string)
        )
        output = self.ai_client.detect_language_entities(
            detect_language_entities_details
        )
        entities = json.loads(str(output.data))["entities"]
        df = pd.DataFrame.from_dict(entities)
        df.columns = ["PII", "Length", "Offset", "Score", "Entity", "Type"]
        return df

    @property
    def language_dominant(self) -> pd.DataFrame:
        detect_dominant_language_details = (
            oci.ai_language.models.DetectDominantLanguageDetails(text=self.string)
        )
        output = self.ai_client.detect_dominant_language(
            detect_dominant_language_details
        )
        languages = json.loads(str(output.data))["languages"]
        df = pd.json_normalize(languages)
        df.columns = ["Code", "Language", "Score"]
        return df

    @property
    def key_phrase(self) -> pd.DataFrame:
        detect_language_key_phrases_details = (
            oci.ai_language.models.DetectLanguageKeyPhrasesDetails(text=self.string)
        )
        output = self.ai_client.detect_language_key_phrases(
            detect_language_key_phrases_details
        )
        key_phrase = json.loads(str(output.data))["key_phrases"]
        df = pd.DataFrame.from_dict(key_phrase)
        df.columns = ["Score", "Text"]
        return df

    @property
    def absa(self) -> pd.DataFrame:
        detect_language_sentiments_details = (
            oci.ai_language.models.DetectLanguageSentimentsDetails(text=self.string)
        )
        output = self.ai_client.detect_language_sentiments(
            detect_language_sentiments_details
        )
        aspects = json.loads(str(output.data))["aspects"]
        df = pd.json_normalize(aspects)
        df.columns = [
            "Length",
            "Offset",
            "Sentiment",
            "Text",
            "Negative",
            "Neutral",
            "Positive",
        ]
        return df

    @property
    def text_classification(self) -> pd.DataFrame:
        detect_language_text_clf_details = (
            oci.ai_language.models.DetectLanguageTextClassificationDetails(
                text=self.string
            )
        )
        output = self.ai_client.detect_language_text_classification(
            detect_language_text_clf_details
        )
        text_classification = json.loads(str(output.data))["text_classification"]
        df = pd.DataFrame.from_dict(text_classification)
        df.columns = ["Label", "Score"]
        return df
