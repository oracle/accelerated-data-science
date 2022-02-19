#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import oci.ai_language

from ads.common import auth as authutil
from ads.common import oci_client


class OCILanguage(object):  # pragma: no cover
    def __init__(self, auth=None):
        auth = auth if auth else authutil.default_signer()
        self.ai_client = oci_client.OCIClientFactory(**auth).ai_language

    @property
    def ner(self):
        detect_language_entities_details = (
            oci.ai_language.models.DetectLanguageEntitiesDetails(text=self.string)
        )
        output = self.ai_client.detect_language_entities(
            detect_language_entities_details
        )
        return json.loads(str(output.data))["entities"]

    @property
    def lang(self):
        detect_dominant_language_details = (
            oci.ai_language.models.DetectDominantLanguageDetails(text=self.string)
        )
        output = self.ai_client.detect_dominant_language(
            detect_dominant_language_details
        )
        return json.loads(str(output.data))

    @property
    def key_phrases(self):
        detect_language_key_phrases_details = (
            oci.ai_language.models.DetectLanguageKeyPhrasesDetails(text=self.string)
        )
        output = self.ai_client.detect_language_key_phrases(
            detect_language_key_phrases_details
        )
        return json.loads(str(output.data))["key_phrases"]

    @property
    def sentiment(self):
        detect_language_sentiments_details = (
            oci.ai_language.models.DetectLanguageSentimentsDetails(text=self.string)
        )
        output = self.ai_client.detect_language_sentiments(
            detect_language_sentiments_details
        )
        return json.loads(str(output.data))["aspects"]

    @property
    def text_classification(self):
        detect_language_text_clf_details = (
            oci.ai_language.models.DetectLanguageTextClassificationDetails(
                text=self.string
            )
        )
        output = self.ai_client.detect_language_text_classification(
            detect_language_text_clf_details
        )
        return json.loads(str(output.data))["text_classification"]
