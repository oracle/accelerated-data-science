#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class Parser:
    @property
    def pos(self):
        raise NotImplementedError()

    @property
    def noun(self):
        raise NotImplementedError()

    @property
    def adjective(self):
        raise NotImplementedError()

    @property
    def adverb(self):
        raise NotImplementedError()

    @property
    def verb(self):
        raise NotImplementedError()

    @property
    def word(self):
        raise NotImplementedError()

    @property
    def sentence(self):
        raise NotImplementedError()

    @property
    def word_count(self):
        raise NotImplementedError()

    @property
    def bigram(self):
        raise NotImplementedError()

    @property
    def trigram(self):
        raise NotImplementedError()
