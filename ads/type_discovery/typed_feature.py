#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import

import json
import re
import copy
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import CountVectorizer

from ads.common import utils, logger
from ads.common.card_identifier import card_identify
from ads.common.utils import JsonConverter
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class TypedFeature(Bunch):
    def __init__(self, name, meta_data):
        assert "type" in meta_data
        self.meta_data = meta_data
        self.meta_data["feature_name"] = name
        super().__init__(**meta_data)

    @staticmethod
    def build(name, series):
        pass

    def __repr__(self):
        d = copy.deepcopy(self.meta_data)
        if "internal" in d:
            d.pop("internal")
        return json.dumps(d, indent=2, cls=JsonConverter)


class ContinuousTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    @runtime_dependency(module="scipy", install_from=OptionalDependency.VIZ)
    def build(name, series):
        series = series.astype("float")

        non_null_series = series.loc[~series.isnull()]
        desc = non_null_series.describe()
        stats = {
            "mode": series.mode().iloc[0],
            "median": desc["50%"],
            "kurtosis": series.kurt(),
            "variance": series.var(),
            "skewness": series.skew(),
            "outlier_percentage": 100
            * np.count_nonzero(np.abs(scipy.stats.zscore(non_null_series) >= 2.999))
            / series.size,
        }
        stats.update({k: v for k, v in desc.items()})

        return ContinuousTypedFeature(
            name,
            {
                "type": "continuous",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": stats,
                "internal": {"sample": series.head(5)},
            },
        )


class DiscreteTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)


class OrdinalTypedFeature(DiscreteTypedFeature):
    def __init__(self, name, meta_data):
        DiscreteTypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        nulls_removed = series.astype("category").loc[~series.isnull()]
        desc = nulls_removed.describe(include="all")
        value_counts = series.value_counts(ascending=False)
        x_min, x_max = np.nanmin(nulls_removed), np.nanmax(nulls_removed)

        stats = {
            "unique percentage": 100 * desc["unique"] / desc["count"],
            "x_min": x_min,
            "x_max": x_max,
            "mode": series.mode().iloc[0],
        }
        stats.update({k: v for k, v in desc.items()})

        return OrdinalTypedFeature(
            name,
            {
                "type": "ordinal",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": stats,
                "internal": {
                    "sample": series.head(5),
                    "unique": desc["unique"],
                    "counts": utils.truncate_series_top_n(
                        value_counts, n=min(16, len(value_counts))
                    ),
                    "high_cardinality": bool(desc["unique"] > 30),
                    "very_high_cardinality": bool(
                        desc["unique"] >= 0.95 * desc["count"]
                    ),
                },
            },
        )


class CategoricalTypedFeature(DiscreteTypedFeature):
    def __init__(self, name, meta_data):
        DiscreteTypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        desc = series.astype("category").loc[~series.isnull()].describe(include="all")
        value_counts = series.value_counts(ascending=False)
        if isinstance(desc["top"], str):
            mode = desc["top"] if len(desc["top"]) < 30 else desc["top"][:24] + "..."
        else:
            mode = desc["top"]

        stats = {
            "unique percentage": 100 * desc["unique"] / desc["count"],
            "mode": mode,
        }
        stats.update({k: v for k, v in desc.items()})

        return CategoricalTypedFeature(
            name,
            {
                "type": "categorical",
                "low_level_type": series.dtype.name,
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "stats": stats,
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "unique": desc["unique"],
                    "counts": utils.truncate_series_top_n(
                        value_counts, n=min(16, len(value_counts))
                    ),
                    "high_cardinality": bool(desc["unique"] > 30),
                    "very_high_cardinality": bool(
                        desc["unique"] >= 0.95 * desc["count"]
                    ),
                },
            },
        )


class IPAddressTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        desc = series.astype("category").loc[~series.isnull()].describe()

        nulls_removed = series.loc[~series.isnull()]
        value_counts = nulls_removed.value_counts()

        return IPAddressTypedFeature(
            name,
            {
                "type": "ipaddress",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": {
                    "unique percentage": 100 * desc["unique"] / desc["count"],
                    "mode": series.mode().iloc[0],
                },
                "internal": {
                    "sample": nulls_removed.sample(n=min(100, nulls_removed.size)),
                    "counts": utils.truncate_series_top_n(
                        value_counts, n=min(16, len(value_counts))
                    ),
                    "unique": desc["unique"],
                },
            },
        )


class PhoneNumberTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        desc = series.astype("category").loc[~series.isnull()].describe()

        nulls_removed = list(series.loc[~series.isnull()])
        pat = re.compile(r"^(\+?\d{1,2}[\s-])?\(?(\d{3})\)?[\s.-]?\d{3}[\s.-]?\d{4}$")
        area_codes = [
            g.group(2)
            for g in filter(None, [pat.match(x) for x in nulls_removed])
            if g and len(g.groups()) >= 2
        ]
        value_counts = pd.Series(area_codes).value_counts()

        # resolve area codes
        return PhoneNumberTypedFeature(
            name,
            {
                "type": "phone",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": {
                    "unique percentage": 100 * desc["unique"] / desc["count"],
                    "mode": series.mode().iloc[0],
                },
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "counts": utils.truncate_series_top_n(
                        value_counts, n=min(16, len(value_counts))
                    ),
                    "unique": desc["unique"],
                },
            },
        )


class GISTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series, samples):
        desc = series.astype("category").loc[~series.isnull()].describe()
        value_counts = series.value_counts(ascending=False)
        return GISTypedFeature(
            name,
            {
                "type": "gis",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": {
                    "observations": desc["count"],
                    "unique percentage": 100 * desc["unique"] / desc["count"]
                    # TODO mid point
                },
                "internal": {"sample": samples, "unique": desc["unique"]},
            },
        )


class AddressTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        nulls_removed = series.loc[~series.isnull()]
        mean_document_length = nulls_removed.str.len().mean()
        sampled_nulls_removed = nulls_removed.sample(n=min(len(nulls_removed), 1000))

        return AddressTypedFeature(
            name,
            {
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "type": "address",
                "low_level_type": series.dtype.name,
                "stats": {
                    "mean_character_count": mean_document_length,
                    "mean_word_length": nulls_removed.str.split().str.len().mean(),
                },
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "word_frequencies": DocumentTypedFeature.vectorization(
                        name, sampled_nulls_removed, mean_document_length
                    ),
                },
            },
        )


class DocumentTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    @runtime_dependency(module="wordcloud", install_from=OptionalDependency.TEXT)
    def corpus_processor(series):
        pat_punct = re.compile(r"[^a-zA-Z]", re.UNICODE + re.MULTILINE)
        pat_tags = re.compile(r"<.*?>", re.UNICODE + re.MULTILINE)
        pat_digits = re.compile(r"(\d|\W)+", re.UNICODE + re.MULTILINE)
        pat_splitter = re.compile(r"\s+", re.UNICODE + re.MULTILINE)
        for doc in series:
            # Remove punctuations, tags,  special characters and digits
            text = pat_punct.sub(" ", doc.lower())

            # remove tags
            text = pat_tags.sub("", text)

            # remove special characters and digits
            text = pat_digits.sub(" ", text)

            from wordcloud import STOPWORDS

            stop_words = STOPWORDS

            # Convert to list from string and return a generator expression
            yield " ".join(
                [
                    x
                    for x in pat_splitter.split(text)
                    if len(x) >= 3 and x not in stop_words
                ]
            )

    @staticmethod
    def sub_vectorization(
        feature_name,
        series,
        min_df=0.0,
        max_df=1.0,
        min_tf=2,
    ):
        start = time()

        v1 = CountVectorizer(
            ngram_range=(1, 1),
            max_features=2000,
            decode_error="ignore",
            strip_accents="unicode",
            stop_words=None,
            min_df=min_df,
            max_df=max_df,
        )
        X1 = v1.fit_transform(DocumentTypedFeature.corpus_processor(series))
        unigrams = {
            k: int(v)
            for k, v in dict(
                zip(v1.get_feature_names_out(), np.asarray(X1.sum(axis=0)).ravel())
            ).items()
        }

        v2 = CountVectorizer(
            ngram_range=(2, 2),
            max_features=2000,
            decode_error="ignore",
            strip_accents="unicode",
            stop_words=None,
            min_df=min_df,
            max_df=max_df,
        )
        X2 = v2.fit_transform(DocumentTypedFeature.corpus_processor(series))
        bigrams = {
            k: int(v)
            for k, v in dict(
                zip(v2.get_feature_names_out(), np.asarray(X2.sum(axis=0)).ravel())
            ).items()
        }

        # drop unigrams that are prefixes of bigrams
        for k, v in bigrams.items():
            # take the key (k), split it to get first word and look for that word in unigrams
            for word in k.split():
                if word in unigrams:
                    # boost the bigram
                    bigrams[k] = v + unigrams[word]
                    del unigrams[word]

        unigrams.update(bigrams)

        ret = {k: v for k, v in unigrams.items() if v >= min_tf}
        return ret if ret else unigrams.items()

    @staticmethod
    def vectorization(feature_name, series, mean_document_length):
        # set the min_df and max_df as functions of the mean_document_length
        min_df = 0.1
        max_df = 0.8

        if mean_document_length > 5000:
            min_df = 0.2
            max_df = 0.7
        try:
            return DocumentTypedFeature.sub_vectorization(
                feature_name, series, min_df=min_df, max_df=max_df
            )
        except ValueError:
            return DocumentTypedFeature.sub_vectorization(
                feature_name, series, min_df=0.0, max_df=1.0
            )

    @staticmethod
    def build(name, series, is_cjk, is_html):
        internal = {"cjk": is_cjk, "html": is_html}

        if is_cjk:
            stats = {"mean_document_character_count": int(series.str.len().mean())}

            internal["sample"] = series.sample(n=min(100, series.size))

        else:
            nulls_removed = series.loc[~series.isnull()]
            if is_html:
                nulls_removed = nulls_removed.replace(
                    to_replace="<[^<]+?>", value=" ", regex=True
                ).replace("\n", " ", regex=True)

            internal["sample"] = nulls_removed.sample(n=min(100, nulls_removed.size))

            mean_document_length = nulls_removed.str.len().mean()
            sample_size = int(
                2e6 // mean_document_length
            )  # vectorize at most 2MB of text

            sampled_nulls_removed = nulls_removed.sample(
                n=min(len(nulls_removed), sample_size)
            )

            stats = {
                "mean_document_character_count": int(series.str.len().mean()),
                "mean_word_length": nulls_removed.str.split().str.len().mean(),
            }

            internal["word_frequencies"] = DocumentTypedFeature.vectorization(
                name, sampled_nulls_removed, mean_document_length
            )

        return DocumentTypedFeature(
            name,
            {
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "type": "document",
                "low_level_type": series.dtype.name,
                "stats": stats,
                "internal": internal,
            },
        )


class ZipcodeTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        max_threshold = 24
        suffix = ""

        series = series.loc[~series.isnull()].astype(str).str.slice(0, 5)

        for i in [5, 4, 3, 2, 1]:
            if series.nunique() <= max_threshold:
                break
            series = series.str[:-1]
            suffix += "*"

        series += suffix

        return ZipcodeTypedFeature(
            name,
            {
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "type": "zipcode",
                "low_level_type": series.dtype.name,
                "stats": {"mode": series.mode().iloc[0]},
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "histogram": series.value_counts(),
                },
            },
        )


class UnknownTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        return UnknownTypedFeature(
            name,
            {
                "type": "unknown",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": {"mode": series.mode().iloc[0]},
                "internal": {"sample": series.sample(n=min(100, series.size))},
            },
        )


class ConstantTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        desc = series.astype("category").loc[~series.isnull()].describe()
        if "top" in desc:
            value = desc["top"]
            value_counts = utils.truncate_series_top_n(series.value_counts())
        else:
            value = "NaN"
            value_counts = {value: len(series)}
        return ConstantTypedFeature(
            name,
            {
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "type": "constant",
                "stats": {"value": value},
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "counts": value_counts,
                },
            },
        )


class CreditCardTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        null_removed_series = series.loc[~series.isnull()]
        desc = null_removed_series.astype("category").describe()
        sampled_series = null_removed_series.sample(
            n=min(1000, len(null_removed_series))
        )

        d_scheme = defaultdict(int)

        # use counting method to build dict of credit card data
        for s in sampled_series:
            scheme = card_identify().identify_issue_network(s)
            d_scheme[scheme] += 1

        return CreditCardTypedFeature(
            name,
            {
                "type": "creditcard",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": {
                    "unique percentage": 100 * desc["unique"] / desc["count"],
                    "mode": desc["top"],
                },
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "counts": dict(d_scheme),
                    "unique": desc["unique"],
                },
            },
        )


class DateTimeTypedFeature(TypedFeature):
    def __init__(self, name, meta_data):
        TypedFeature.__init__(self, name, meta_data)

    @staticmethod
    def build(name, series):
        desc = series.loc[~series.isnull()].describe()

        return DateTimeTypedFeature(
            name,
            {
                "type": "datetime",
                "missing_percentage": 100 * series.isna().sum() / series.size,
                "low_level_type": series.dtype.name,
                "stats": {
                    "unique percentage": 100 * desc["unique"] / desc["count"],
                    "first": desc["first"],
                    "last": desc["last"],
                    "mode": desc["top"],
                },
                "internal": {
                    "sample": series.sample(n=min(100, series.size)),
                    "unique": desc["unique"],
                },
            },
        )
