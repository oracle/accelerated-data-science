#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import nltk

from ads.feature_engineering.feature_type.adsstring.oci_language import OCILanguage
from ads.feature_engineering.feature_type.adsstring.string import ADSString


class TestIntegration:
    def setup_class(self):
        nltk.download("punkt", download_dir=os.environ["CONDA_PREFIX"] + "/nltk")
        nltk.download(
            "averaged_perceptron_tagger",
            download_dir=os.environ["CONDA_PREFIX"] + "/nltk",
        )

    def test_adsmixin_methods(self):
        test_text = """
            Lawrence Joseph Ellison (born August 17, 1944) is an American business magnate,
            investor, and philanthropist who is a co-founder, the executive chairman and
            chief technology officer (CTO) of Oracle Corporation. As of October 2019, he was
            listed by Forbes magazine as the fourth-wealthiest person in the United States
            and as the sixth-wealthiest in the world, with a fortune of $69.1 billion,
            increased from $54.5 billion in 2018.[4] He is also the owner of the 41st
            largest island in the United States, Lanai in the Hawaiian Islands with a
            population of just over 3000
        """.strip()
        ADSString.nlp_backend(backend="nltk")
        s = ADSString(test_text)

        assert len(s.word) == 82
        assert len(s.adjective) == 5
        assert len(s.noun) == 33
        assert len(s.adverb) == 2
        assert len(s.verb) == 8
        assert len(s.sentence) == 3
        word_count = s.word_count.values.tolist()
        assert len(word_count) == 53
        assert word_count[0] == ["the", 9]
        assert s.pos.values.tolist()[0] == ["Lawrence", "NNP"]

    def test_oci_language_integration(self):
        test_text = """This was an absolutely terrible movie. Don't be lured in
        by Christopher Walken or Michael Ironside. Both are great actors,
        but this must simply be their worst role in history. Even their
        great acting could not redeem this movie's ridiculous storyline.
        This movie is an early nineties US propaganda piece. The most
        pathetic scenes were those when the rebels were making
        their cases for revolutions. Maria Conchita Alonso appeared phony,
        and her pseudo-love affair with Walken was nothing but a pathetic emotional plug
        in a movie that was devoid of any real meaning. I
        am disappointed that there are movies like this, ruining actor's
        like Christopher Walken's good name. I could barely sit through it.
    """
        ADSString.plugin_register(OCILanguage)
        s = ADSString(test_text)
        from collections import Counter

        assert s.text_classification[0]["label"].lower() == "entertainment/movies"
        assert s.key_phrase[0]["text"].lower() == "movie"

        assert (
            Counter(v["sentiment"] for v in s.absa).most_common(1)[0][0] == "Negative"
        )

        entities = s.ner
        assert entities[0]["length"] == 18
        assert entities[0]["offset"] == 68
        assert entities[0]["score"] > 0.9
        assert entities[0]["text"] == "Christopher Walken"
        assert entities[0]["type"] == "PERSON"

        languages = s.language_dominant["languages"][0]

        assert languages["code"] == "en"
        assert languages["name"] == "English"
        assert languages["score"] > 0.9

        test_text = test_text.lower()

        for item in s.key_phrase:
            assert item["text"].lower() in test_text
