.. _string-nlp_parse:

NLP Parse
*********

``ADSString`` also supports NLP parsing and is backed by `Natural Language Toolkit (NLTK) <https://www.nltk.org/>`__ or `spaCy <https://spacy.io/>`__.  Unless otherwise specified, NLTK is used by default. You can extract properties, such as nouns, adjectives, word counts, parts of speech tags, and so on from text with NLP.

The ``ADSString`` class can have one backend enabled at a time. What properties are available depends on the backend, as do the results of calling the property. The following examples provide an overview of the available parsers, and how to use them. Generally, the parser supports the ``adjective``, ``adverb``, ``bigram``, ``noun``, ``pos``, ``sentence``, ``trigram``, ``verb``, ``word``, and ``word_count`` base properties. Parsers can support additional parsers.

NLTK
====

The Natural Language Toolkit (`NLTK <https://www.nltk.org/>`__) is a powerful platform for processing human language data. It supports all the base properties and in addition ``stem`` and ``token``. The ``stem`` property returns a list of all the stemmed tokens. It reduces a token to its word stem that affixes to suffixes and prefixes, or to the roots of words that is the lemma. The ``token`` property is similar to the ``word`` property, except it returns non-alphanumeric tokens and doesn’t force tokens to be lowercase.

The following example use a sample of text about Larry Ellison to demonstrate the use of the NLTK properties.

.. code-block:: python3

    test_text = """
                Lawrence Joseph Ellison (born August 17, 1944) is an American business magnate,
                investor, and philanthropist who is a co-founder, the executive chairman and
                chief technology officer (CTO) of Oracle Corporation. As of October 2019, he was
                listed by Forbes magazine as the fourth-wealthiest person in the United States
                and as the sixth-wealthiest in the world, with a fortune of $69.1 billion,
                increased from $54.5 billion in 2018.[4] He is also the owner of the 41st
                largest island in the United States, Lanai in the Hawaiian Islands with a
                population of just over 3000.
            """.strip()
    ADSString.nlp_backend("nltk")
    s = ADSString(test_text)

.. code-block:: python3

    s.noun[1:5]

.. parsed-literal::

    ['Joseph', 'Ellison', 'August', 'business']

.. code-block:: python3

    s.adjective

.. parsed-literal::

    ['American', 'chief', 'fourth-wealthiest', 'largest', 'Hawaiian']

.. code-block:: python3

    s.word[1:5]

.. parsed-literal::

    ['joseph', 'ellison', 'born', 'august']

By taking the difference between ``token`` and ``word``, the token set contains non-alphanumeric tokes, and also the uppercase version of words.

.. code-block:: python3

    list(set(s.token) - set(s.word))[1:5]

.. parsed-literal::

    ['Oracle', '1944', '41st', 'fourth-wealthiest']

The ``stem`` property takes the list of words and stems them. It produces morphological variations of a word’s root form. The following example stems some words, and shows some of the stemmed words that were changed.

.. code-block:: python3

    list(set(s.stem) - set(s.word))[1:5]

.. parsed-literal::

    ['fortun', 'technolog', 'increas', 'popul']

Part of Speech Tags
-------------------

Part of speech (POS) is a category in which a word is assigned based on its syntactic function. POS depends on the language. For English, the most common POS are adjective, adverb, conjunction, determiner, interjection, noun, preposition, pronoun, and verb. However, each POS system has its own set of POS tags that vary based on their respective training set. The NLTK parsers produce the following POS tags:

.. list-table:: Parts of Speech Tags
   :widths: auto
   :header-rows: 0

   * * ``CC``: coordinating conjunction
     * ``CD``: cardinal digit
   * * ``DT``: determiner
     * ``EX``: existential there; like "there is"; "there exists"
   * * ``FW``: foreign word
     * ``IN``: preposition/subordinating conjunction
   * * ``JJ``: adjective; "big"
     * ``JJR``: adjective, comparative; "bigger"
   * * ``JJS``: adjective, superlative; "biggest"
     * ``LS``: list marker 1)
   * * ``MD``: modal could, will
     * ``NN``: noun, singular; "desk"
   * * ``NNS``: noun plural; "desks"
     * ``NNP``: proper noun, singular; "Harrison"
   * * ``NNPS``: proper noun, plural; "Americans"
     * ``PDT``: predeterminer; "all the kids"
   * * ``POS``: possessive ending; "parent’s"
     * ``PRP``: personal pronoun; I, he, she
   * * ``PRP$``: possessive pronoun; my, his, hers
     * ``RB``: adverb; very, silently
   * * ``RBR``: adverb; comparative better
     * ``RBS``: adverb; superlative best
   * * ``RP``: particle; give up
     * ``TO``: to go; “to” the store.
   * * ``UH``: interjection; errrrrrrrm
     * ``VB``: verb, base form; take
   * * ``VBD``: verb, past tense; took
     * ``VBG``: verb, gerund/present participle; taking
   * * ``VBN``: verb, past participle; taken
     * ``VBP``: verb, singular present; non-3d take
   * * ``VBZ``: verb, 3rd person singular present; takes
     * ``WDT``: wh-determiner; which
   * * ``WP``: wh-pronoun; who, what
     * ``WP$``: possessive wh-pronoun; whose
   * * ``WRB``: wh-adverb; where, when
     *

.. code-block:: python3

    s.pos[1:5]

.. image:: figures/nltk_pos.png
    :alt: Listing of Part-of-Speech tags

spaCy
=====

spaCy is in an advanced NLP toolkit. It helps you understand what the words mean in context, and who is doing what to whom. It helps you determine what companies and products are mentioned in a document. The spaCy backend is used to parses the ``adjective``, ``adverb``, ``bigram``, ``noun``, ``pos``, ``sentence``, ``trigram``, ``verb``, ``word``, and ``word_count`` base properties. It also supports the following additional properties:

* ``entity``: All entities in the text.
* ``entity_artwork``: The titles of books, songs, and so on.
* ``entity_location``: Locations, facilities, and geopolitical entities, such as countries, cities, and states.
* ``entity_organization``: Companies, agencies, and institutions.
* ``entity_person``: Fictional and real people.
* ``entity_product``: Product names and so on.
* ``lemmas``: A rule-based estimation of the roots of a word.
* ``tokens``: The base tokens of the tokenization process. This is similar to ``word``, but it includes non-alphanumeric values and the word case is preserved.

If the ``spacy`` module is installed ,you can change the NLP backend using the ``ADSString.nlp_backend('spacy')`` command.

.. code-block:: python3

    ADSString.nlp_backend("spacy")
    s = ADSString(test_text)

.. code-block:: python3

    s.noun[1:5]

.. parsed-literal::

    ['magnate', 'investor', 'philanthropist', 'co']

.. code-block:: python3

    s.adjective

.. parsed-literal::

    ['American', 'executive', 'chief', 'fourth', 'wealthiest', 'largest']

.. code-block:: python3

    s.word[1:5]

.. parsed-literal::

    ['Joseph', 'Ellison', 'born', 'August']

You can identify all the locations that are mentioned in the text.

.. code-block:: python3

    s.entity_location

.. parsed-literal::

    ['the United States', 'the Hawaiian Islands']

Also, the organizations that were mentioned.

.. code-block:: python3

    s.entity_organization

.. parsed-literal::

    ['CTO', 'Oracle Corporation', 'Forbes', 'Lanai']

Part of Speech Tags
-------------------

The POS tagger in `spaCy <https://spacy.io/>`__ uses a smaller number of categories. For example, spaCy has the ``ADJ`` POS for all adjectives, while NLTK has ``JJ`` to mean an adjective. ``JJR`` refers to a comparative adjective, and ``JJS`` refers to a superlative adjective. For fine grain analysis of different parts of speech, NLTK is the preferred backend. However, spaCy’s reduced category set tends to produce fewer errors,at the cost of not being as specific.

The spaCy parsers produce the following POS tags:

* ``ADJ``: adjective; big, old, green, incomprehensible, first
* ``ADP``: adposition; in, to, during
* ``ADV``: adverb; very, tomorrow, down, where, there
* ``AUX``: auxiliary; is, has (done), will (do), should (do)
* ``CONJ``: conjunction; and, or, but
* ``CCONJ``: coordinating conjunction; and, or, but
* ``DET``: determiner; a, an, the
* ``INTJ``: interjection; psst, ouch, bravo, hello
* ``NOUN``: noun; girl, cat, tree, air, beauty
* ``NUM``: numeral; 1, 2017, one, seventy-seven, IV, MMXIV
* ``PART``: particle; ’s, not,
* ``PRON``: pronoun; I, you, he, she, myself, themselves, somebody
* ``PROPN``: proper noun; Mary, John, London, NATO, HBO
* ``PUNCT``: punctuation; ., (, ), ?
* ``SCONJ``: subordinating conjunction; if, while, that
* ``SYM``: symbol; $, %, §, ©, +, −, ×, ÷, =, :), 😝
* ``VERB``: verb; run, runs, running, eat, ate, eating
* ``X``: other; sfpksdpsxmsa
* ``SPACE``: space

.. code-block:: python3

    s.pos[1:5]

.. image:: figures/spacy_pos.png
    :alt: Listing of Part-of-Speech tags

