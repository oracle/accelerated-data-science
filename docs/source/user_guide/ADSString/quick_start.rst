.. _string_quick_start:

Quick Start
***********

NLP Parse
=========

The following example parses a text corpus using the `NTLK <https://www.nltk.org/>`__ and `spaCy <https://spacy.io/>`__ engines.

.. code-block:: python3

    from ads.feature_engineering.adsstring.string import ADSString
    
    
    s = ADSString("""
        Lawrence Joseph Ellison (born August 17, 1944) is an American business magnate,
        investor, and philanthropist who is a co-founder, the executive chairman and
        chief technology officer (CTO) of Oracle Corporation. As of October 2019, he was
        listed by Forbes magazine as the fourth-wealthiest person in the United States
        and as the sixth-wealthiest in the world, with a fortune of $69.1 billion,
        increased from $54.5 billion in 2018.[4] He is also the owner of the 41st
        largest island in the United States, Lanai in the Hawaiian Islands with a
        population of just over 3000.
        """.strip())

    # NLTK
    ADSString.nlp_backend("nltk")
    noun = s.noun
    adj = s.adjective
    pos = s.pos # Parts of Speech

    # spaCy
    ADSString.nlp_backend("spacy")
    noun = s.noun
    adj = adjective
    pos = s.pos # Parts of Speech 

Plugin
======

Custom Plugin
-------------

This example demonstrates how to create a custom plugin that will take a string, detect the credit card numbers, and return a list of the  last four digits of the credit card number.

.. code-block:: python3

    from ads.feature_engineering.adsstring.string import ADSString
    
    class CreditCardLast4:
        @property
        def credit_card_last_4(self):
            return [x[len(x)-4:len(x)] for x in ADSString(self.string).credit_card]
        
    ADSString.plugin_register(CreditCardLast4)
    
    creditcard_numbers = "I purchased the gift on this card 4532640527811543 and the dinner on 340984902710890"
    s = ADSString(creditcard_numbers)
    s.credit_card_last_4


OCI Language Services Plugin
----------------------------

This example uses the `OCI Language service <https://docs.oracle.com/iaas/language/using/overview.htm>`__ to perform an aspect-based sentiment analysis, language detection, key phrase extraction, and a named entity recognition.

.. code-block:: python3

    from ads.feature_engineering.adsstring.oci_language import OCILanguage
    from ads.feature_engineering.adsstring.string import ADSString
    
    ADSString.plugin_register(OCILanguage)
    
    s = ADSString("""
        Lawrence Joseph Ellison (born August 17, 1944) is an American business magnate,
        investor, and philanthropist who is a co-founder, the executive chairman and
        chief technology officer (CTO) of Oracle Corporation. As of October 2019, he was
        listed by Forbes magazine as the fourth-wealthiest person in the United States
        and as the sixth-wealthiest in the world, with a fortune of $69.1 billion,
        increased from $54.5 billion in 2018.[4] He is also the owner of the 41st
        largest island in the United States, Lanai in the Hawaiian Islands with a
        population of just over 3000.
        """.strip())
    
    # Aspect-Based Sentiment Analysis
    df_sentiment = s.absa
    
    # Key Phrase Extraction
    key_phrase = s.key_phrase
    
    # Language Detection
    language = s.language_dominant
    
    # Named Entity Recognition
    named_entity = s.ner

    # Text Classification
    classification = s.text_classification

RegEx Match
===========

In this example, the dates and prices are extracted from the text using regular expression matching.

.. code-block:: python3

    from ads.feature_engineering.adsstring.string import ADSString
    
    s = ADSString("""
        Lawrence Joseph Ellison (born August 17, 1944) is an American business magnate,
        investor, and philanthropist who is a co-founder, the executive chairman and
        chief technology officer (CTO) of Oracle Corporation. As of October 2019, he was
        listed by Forbes magazine as the fourth-wealthiest person in the United States
        and as the sixth-wealthiest in the world, with a fortune of $69.1 billion,
        increased from $54.5 billion in 2018.[4] He is also the owner of the 41st
        largest island in the United States, Lanai in the Hawaiian Islands with a
        population of just over 3000.
    """.strip())
    
    dates = s.date
    prices = s.price


