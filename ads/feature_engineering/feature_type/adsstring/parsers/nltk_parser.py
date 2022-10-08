#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import functools
import nltk
import pandas as pd
from nltk import FreqDist, ngrams, pos_tag
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from ads.feature_engineering.feature_type.adsstring.parsers.base import Parser
from typing import List

if "CONDA_PREFIX" in os.environ:
    nltk.data.path.append(f"{os.environ['CONDA_PREFIX']}/nltk")


class NLTKParser(Parser):

    stemmers = {}  # language -> stemmer

    def __init__(self):
        if not self.language in NLTKParser.stemmers:
            NLTKParser.stemmers[self.language] = SnowballStemmer(self.language)

    @property
    @functools.lru_cache()
    def pos(self) -> pd.DataFrame:
        return pd.DataFrame(data=self._pos(), columns=["Word", "Label"])

    @functools.lru_cache()
    def _pos(self):
        return pos_tag(word_tokenize(self.string))

    @property
    @functools.lru_cache()
    def adjective(self) -> List[str]:
        return [word for (word, pos) in self._pos() if pos.startswith("JJ")]

    @property
    @functools.lru_cache()
    def noun(self) -> List[str]:
        return [word for (word, pos) in self._pos() if pos.startswith("NN")]

    @property
    @functools.lru_cache()
    def adverb(self) -> List[str]:
        return [word for (word, pos) in self._pos() if pos.startswith("RB")]

    @property
    @functools.lru_cache()
    def verb(self) -> List[str]:
        return [word for (word, pos) in self._pos() if pos.startswith("VB")]

    @property
    @functools.lru_cache()
    def sentence(self) -> List[str]:
        return sent_tokenize(self.string)

    @property
    @functools.lru_cache()
    def token(self) -> List[str]:
        return word_tokenize(self.string)

    @property
    def word(self) -> List[str]:
        return [word.lower() for word in self.token if word.isalpha()]

    @property
    @functools.lru_cache()
    def word_count(self) -> pd.DataFrame:
        word_count = list(FreqDist(self.word).items())
        df = pd.DataFrame(data=word_count, columns=["Word", "Count"])
        return df.sort_values(
            ["Count", "Word"], ascending=[False, True], ignore_index=True
        )

    @property
    @functools.lru_cache()
    def bigram(self) -> pd.DataFrame:
        res = list(ngrams(self.word, 2))
        return pd.DataFrame(data=res, columns=["Word 1", "Word 2"])

    @property
    @functools.lru_cache()
    def trigram(self) -> pd.DataFrame:
        res = list(ngrams(self.word, 3))
        return pd.DataFrame(data=res, columns=["Word 1", "Word 2", "Word 3"])

    @property
    @functools.lru_cache()
    def stem(self) -> List:
        return [NLTKParser.stemmers[self.language].stem(w) for w in self.word]
