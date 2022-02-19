#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import functools

from typing import Any, Callable
from ads.feature_engineering.adsstring.common_regex_mixin import (
    CommonRegexMixin,
)
from ads.feature_engineering.adsstring.oci_language import OCILanguage


def to_ads_string(func: Callable) -> Callable:
    """Decorator that converts output of a function to `ADSString` if it returns a string.

    Parameters
    ----------
    func : Callable
        function to decorate

    Returns
    -------
    Callable
        decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        if isinstance(out, str):
            return ADSString(out)
        else:
            return out

    return wrapper


def wrap_output_string(decorator: Callable) -> Callable:
    """Class decorator that applies a decorator to all methods of a class.

    Parameters
    ----------
    decorator : Callable
        decorator to apply

    Returns
    -------
    Callable
        class decorator
    """

    def decorate(cls):
        for attr in dir(cls):
            if (
                callable(getattr(cls, attr))
                and not attr.startswith("__")
                and not attr.startswith("_")
            ):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


@wrap_output_string(to_ads_string)
class ADSString(str, CommonRegexMixin):
    """Defines an enhanced string class for the purporse of performing NLP tasks.
    Its functionalities can be extended by registering plugins.

    Attributes
    ----------
    plugins: List
        list of plugins that add functionalities to the class.
    string: str
        plain string

    Example
    -------
    >>> ADSString.set_nlp_backend('nltk')
    >>> s = ADSString("Walking my dog on a breezy day is the best.")
    >>> s.lower() # regular string methods still work
    >>> s.replace("a", "e")
    >>> s.nouns
    >>> s.parts_of_speech
    >>> s = ADSString("get in touch with my associate at john.smith@gmail.com to schedule")
    >>> s.emails
    >>> ADSString.register_plugin(OCILanguage)
    >>> s = ADSString("This movie is awesome.")
    >>> s.sentiment
    """

    plugins = []
    language_model_cache = dict()

    def __init__(self, text: str, language="english") -> None:
        """Initialze the class and register plugins.

        Parameters
        ----------
        text : str
            input text
        language : str, optional
            language of the text, by default "english".

        Raises
        ------
        TypeError
            input text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        if isinstance(text, ADSString):
            self.raw = text.string
        else:
            self.raw = text
        self._string = self.raw.strip()
        self.language = language

        if not hasattr(ADSString, "bases"):
            # remember the bases before adding any plugins
            setattr(ADSString, "bases", ADSString.__bases__)
        else:
            # reset bases before installing plugins
            ADSString.__bases__ = ADSString.bases

        # adding default nlp plugin
        ADSString.plugins.insert(0, OCILanguage)

        # remove duplicates and preserve order
        _plugins = []
        for plg in ADSString.plugins[::-1]:
            if plg not in _plugins:
                _plugins.insert(0, plg)

        for cls in _plugins:
            self.__class__.__bases__ = (cls,) + self.__class__.__bases__
            super(cls, self).__init__()
        super(ADSString, self).__init__()

    @property
    def string(self):
        return self._string

    @classmethod
    def set_nlp_backend(cls, backend: str = "nltk") -> None:
        """Set backend for extracting NLP related properties.

        Parameters
        ----------
        backend : str, optional
            name of backend, by default 'nltk'.

        Raises
        ------
        ModuleNotFoundError
            module corresponding to backend is not found.
        ValueError
            input backend is invalid.

        Returns
        -------
        None
        """
        if backend == "spacy":
            try:
                import spacy
            except:
                raise ModuleNotFoundError("spacy must be installed.")
            from ads.feature_engineering.adsstring.parsers.spacy_parser import (
                SpacyParser,
            )

            cls.register_plugin(SpacyParser)
        elif backend == "nltk":
            try:
                import nltk
            except:
                raise ModuleNotFoundError("nltk must be installed.")
            from ads.feature_engineering.adsstring.parsers.nltk_parser import (
                NLTKParser,
            )

            cls.register_plugin(NLTKParser)
        else:
            raise ValueError(
                "Currently only `nltk` and `spacy` are supported. Default uses `nltk`."
            )

    @classmethod
    def clear_plugins(cls) -> None:
        """Clears plugins."""
        cls.plugins.clear()

    @classmethod
    def register_plugin(cls, plugin: Any) -> None:
        """Register a plugin

        Parameters
        ----------
        plugin : Any
            plugin to register

        Returns
        -------
        None
        """
        cls.plugins.append(plugin)

    @classmethod
    def help(cls) -> None:
        """List available properties."""
        props = [
            attr
            for attr in dir(cls)
            if not attr.startswith("__") and not attr.startswith("_")
        ]
        print(f"{cls.__name__}::Available properties: {', '.join(props)}")
