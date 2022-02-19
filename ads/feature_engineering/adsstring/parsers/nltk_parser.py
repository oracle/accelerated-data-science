#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import functools
import nltk
from nltk import FreqDist, ngrams, pos_tag
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from ads.feature_engineering.adsstring.parsers.base import Parser

if "CONDA_PREFIX" in os.environ:
    nltk.data.path.append(f"{os.environ['CONDA_PREFIX']}/nltk")


class NLTKParser(Parser):

    stemmers = {}  # language -> stemmer

    def __init__(self):
        if not self.language in NLTKParser.stemmers:
            NLTKParser.stemmers[self.language] = SnowballStemmer(self.language)

    @property
    @functools.lru_cache()
    def parts_of_speech(self):
        return pos_tag(word_tokenize(self.string))

    @property
    @functools.lru_cache()
    def adjectives(self):
        return [word for (word, pos) in self.parts_of_speech if pos.startswith("JJ")]

    @property
    @functools.lru_cache()
    def nouns(self):
        return [word for (word, pos) in self.parts_of_speech if pos.startswith("NN")]

    @property
    @functools.lru_cache()
    def adverbs(self):
        return [word for (word, pos) in self.parts_of_speech if pos.startswith("RB")]

    @property
    @functools.lru_cache()
    def verbs(self):
        return [word for (word, pos) in self.parts_of_speech if pos.startswith("VB")]

    @property
    @functools.lru_cache()
    def sentences(self):
        return sent_tokenize(self.string)

    @property
    @functools.lru_cache()
    def tokens(self):
        return word_tokenize(self.string)

    @property
    def words(self):
        return [word.lower() for word in self.tokens if word.isalpha()]

    @property
    @functools.lru_cache()
    def histogram(self):
        return list(FreqDist(self.words).items())

    @property
    @functools.lru_cache()
    def bigrams(self):
        return list(ngrams(self.words, 2))

    @property
    @functools.lru_cache()
    def trigrams(self):
        return list(ngrams(self.words, 3))

    @property
    @functools.lru_cache()
    def stems(self):
        return [NLTKParser.stemmers[self.language].stem(w) for w in self.words]
