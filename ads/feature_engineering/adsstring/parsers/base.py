#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class Parser:
    @property
    def parts_of_speech(self):
        raise NotImplementedError()

    @property
    def nouns(self):
        raise NotImplementedError()

    @property
    def adjectives(self):
        raise NotImplementedError()

    @property
    def adverbs(self):
        raise NotImplementedError()

    @property
    def verbs(self):
        raise NotImplementedError()

    @property
    def words(self):
        raise NotImplementedError()

    @property
    def sentences(self):
        raise NotImplementedError()

    @property
    def histogram(self):
        raise NotImplementedError()

    @property
    def bigrams(self):
        raise NotImplementedError()

    @property
    def trigrams(self):
        raise NotImplementedError()
