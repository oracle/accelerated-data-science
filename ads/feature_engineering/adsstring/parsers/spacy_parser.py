#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import itertools
from collections import Counter
from typing import List, Sequence, Tuple
import os
import functools

import spacy

from ads.feature_engineering.adsstring.parsers.base import Parser


# This module is only used when user installed spacy. spacy has conflicts with oci-cli.
class SpacyParser(Parser):  # pragma: no cover
    def __init__(self):
        try:
            # for ADS conda packs, spacy files are located under $CONDA_PREFIX/spacy
            self.nlp = spacy.load(f"{os.environ['CONDA_PREFIX']}/spacy")
        except OSError:
            try:
                # if not inside a conda pack, trying loading first
                # if failed download to default path
                self.nlp = spacy.load("en_core_web_sm")
            except:
                from spacy.cli import download

                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

    @property
    @functools.lru_cache()
    def _parsed_nlp_obj(self):
        return self.nlp(self.string)

    @property
    @functools.lru_cache()
    def parts_of_speech(self) -> List[Tuple[str, str]]:
        return [
            (token.text, token.pos_)
            for token in self._parsed_nlp_obj
            if not token.is_space
        ]

    def _pos_tokens(self, pos: str) -> List[str]:
        return _f7(
            [
                token.text
                for token in self._parsed_nlp_obj
                if (
                    not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and token.pos_ == pos
                )
            ]
        )

    @property
    @functools.lru_cache()
    def nouns(self) -> List[str]:
        return self._pos_tokens("NOUN")

    @property
    @functools.lru_cache()
    def adjectives(self) -> List[str]:
        return self._pos_tokens("ADJ")

    @property
    @functools.lru_cache()
    def adverbs(self) -> List[str]:
        return self._pos_tokens("ADV")

    @property
    @functools.lru_cache()
    def verbs(self) -> List[str]:
        return self._pos_tokens("VERB")

    @property
    @functools.lru_cache()
    def noun_phrases(self) -> List[str]:
        np = []
        for sentence in self.sentences:
            for chunk in self.nlp(sentence).noun_chunks:
                if chunk.text.strip():
                    np.append(chunk.text.strip())

        return np

    @property
    @functools.lru_cache()
    def extract_entities(self) -> List[Tuple[str, str]]:
        entities = []
        for sentence in self.sentences:
            entities.extend([(ent.text, ent.label_) for ent in self.nlp(sentence).ents])

        return _f7(entities)

    def _entity_tokens(self, entity_label: str) -> List[str]:
        return [
            token for (token, label) in self.extract_entities if label == entity_label
        ]

    @property
    @functools.lru_cache()
    def people_entities(self) -> List[str]:
        # People, including fictional.
        return self._entity_tokens("PERSON")

    @property
    @functools.lru_cache()
    def location_entities(self) -> List[str]:
        # Location entity, i.e. countries, cities, states, mountain ranges, bodies of water, buildings, airports, highways, bridges
        return (
            self._entity_tokens("GPE")
            + self._entity_tokens("LOC")
            + self._entity_tokens("FAC")
        )

    @property
    @functools.lru_cache()
    def organization_entities(self) -> List[str]:
        # Companies, agencies, institutions.
        return self._entity_tokens("ORG")

    @property
    @functools.lru_cache()
    def artwork_entities(self) -> List[str]:
        # title of books, songs, etc
        return self._entity_tokens("WORK_OF_ART")

    @property
    @functools.lru_cache()
    def product_entities(self) -> List[str]:
        # product names, etc
        return self._entity_tokens("PRODUCT")

    @property
    @functools.lru_cache()
    def sentences(self) -> List[str]:
        return [sent.text.strip() for sent in self._parsed_nlp_obj.sents]

    @property
    @functools.lru_cache()
    def tokens(self) -> List[str]:
        return [token.text for token in self._parsed_nlp_obj]

    def _words_by_sentence(self) -> List[str]:
        x = []
        for sentence in self.sentences:
            x.append([token.text for token in self.nlp(sentence) if token.is_alpha])

        return x

    @property
    @functools.lru_cache()
    def words(self):
        return list(itertools.chain(*self._words_by_sentence()))

    def _ngrams(self, tokens, n):
        return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]

    @property
    @functools.lru_cache()
    def bigrams(self):
        return list(
            itertools.chain(
                *[
                    self._ngrams(sentence_words, 2)
                    for sentence_words in self._words_by_sentence()
                ]
            )
        )

    @property
    @functools.lru_cache()
    def trigrams(self):
        return list(
            itertools.chain(
                *[
                    self._ngrams(sentence_words, 3)
                    for sentence_words in self._words_by_sentence()
                ]
            )
        )

    @property
    @functools.lru_cache()
    def lemmas(self):
        return [token.lemma_ for token in self._parsed_nlp_obj]

    @property
    @functools.lru_cache()
    def histogram(self):
        return Counter([word.lower() for word in self.words]).most_common()


def _f7(seq: Sequence) -> List:
    """order preserving de-duplicate sequence"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
