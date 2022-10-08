#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import itertools
from collections import Counter
from typing import List, Sequence, Tuple
import os
import functools
import pandas as pd

from ads.feature_engineering.feature_type.adsstring.parsers.base import Parser
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


# This module is only used when user installed spacy. spacy has conflicts with oci-cli.
class SpacyParser(Parser):  # pragma: no cover
    @runtime_dependency(module="spacy", install_from=OptionalDependency.TEXT)
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
    def pos(self) -> pd.DataFrame:
        pos = (
            (token.text, token.pos_)
            for token in self._parsed_nlp_obj
            if not token.is_space
        )
        return pd.DataFrame(data=pos, columns=["Word", "Label"])

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
    def noun(self) -> List[str]:
        return self._pos_tokens("NOUN")

    @property
    @functools.lru_cache()
    def adjective(self) -> List[str]:
        return self._pos_tokens("ADJ")

    @property
    @functools.lru_cache()
    def adverb(self) -> List[str]:
        return self._pos_tokens("ADV")

    @property
    @functools.lru_cache()
    def verb(self) -> List[str]:
        return self._pos_tokens("VERB")

    @property
    @functools.lru_cache()
    def noun_phrase(self) -> List[str]:
        np = []
        for sentence in self.sentence:
            for chunk in self.nlp(sentence).noun_chunks:
                if chunk.text.strip():
                    np.append(chunk.text.strip())

        return np

    @property
    @functools.lru_cache()
    def entity_extract(self) -> pd.DataFrame:
        return pd.DataFrame(data=self._entity_extract(), columns=["Entity", "Label"])

    @functools.lru_cache()
    def _entity_extract(self) -> List[Tuple[str, str]]:
        entities = []
        for sentence in self.sentence:
            entities.extend([(ent.text, ent.label_) for ent in self.nlp(sentence).ents])

        return _f7(entities)

    def _entity_tokens(self, entity_label: str) -> List[str]:
        return [
            token for (token, label) in self._entity_extract() if label == entity_label
        ]

    @property
    @functools.lru_cache()
    def entity_people(self) -> List[str]:
        # People, including fictional.
        return self._entity_tokens("PERSON")

    @property
    @functools.lru_cache()
    def entity_location(self) -> List[str]:
        # Location entity, i.e. countries, cities, states, mountain ranges, bodies of water, buildings, airports, highways, bridges
        return (
            self._entity_tokens("GPE")
            + self._entity_tokens("LOC")
            + self._entity_tokens("FAC")
        )

    @property
    @functools.lru_cache()
    def entity_organization(self) -> List[str]:
        # Companies, agencies, institutions.
        return self._entity_tokens("ORG")

    @property
    @functools.lru_cache()
    def entity_artwork(self) -> List[str]:
        # title of books, songs, etc
        return self._entity_tokens("WORK_OF_ART")

    @property
    @functools.lru_cache()
    def entity_product(self) -> List[str]:
        # product names, etc
        return self._entity_tokens("PRODUCT")

    @property
    @functools.lru_cache()
    def sentence(self) -> List[str]:
        return [sent.text.strip() for sent in self._parsed_nlp_obj.sents]

    @property
    @functools.lru_cache()
    def token(self) -> List[str]:
        return [token.text for token in self._parsed_nlp_obj]

    def _words_by_sentence(self) -> List[str]:
        x = []
        for sentence in self.sentence:
            x.append([token.text for token in self.nlp(sentence) if token.is_alpha])

        return x

    @property
    @functools.lru_cache()
    def word(self) -> List[str]:
        return list(itertools.chain(*self._words_by_sentence()))

    def _ngrams(self, tokens, n):
        return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]

    @property
    @functools.lru_cache()
    def bigram(self) -> pd.DataFrame:
        bigram = list(
            itertools.chain(
                *[
                    self._ngrams(sentence_words, 2)
                    for sentence_words in self._words_by_sentence()
                ]
            )
        )
        return pd.DataFrame(data=bigram, columns=["Word 1", "Word 2"])

    @property
    @functools.lru_cache()
    def trigram(self) -> pd.DataFrame:
        trigram = list(
            itertools.chain(
                *[
                    self._ngrams(sentence_words, 3)
                    for sentence_words in self._words_by_sentence()
                ]
            )
        )
        return pd.DataFrame(data=trigram, columns=["Word 1", "Word 2", "Word 3"])

    @property
    @functools.lru_cache()
    def lemma(self) -> List[str]:
        return [token.lemma_ for token in self._parsed_nlp_obj]

    @property
    @functools.lru_cache()
    def word_count(self) -> pd.DataFrame:
        word_count = Counter([word.lower() for word in self.word]).most_common()
        df = pd.DataFrame(data=word_count, columns=["Word", "Count"])
        return df.sort_values(
            ["Count", "Word"], ascending=[False, True], ignore_index=True
        )


def _f7(seq: Sequence) -> List:
    """order preserving de-duplicate sequence"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
