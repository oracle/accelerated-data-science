#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import operator
import types
import uuid
from functools import partial
from sklearn.preprocessing import FunctionTransformer
from ads.common.decorator.deprecate import deprecated

try:
    import dask
    from dask.dataframe.accessor import Accessor
    from dask.utils import OperatorMethodMixin
except ImportError as e:
    raise ModuleNotFoundError("Install dask to use the Dask Series class.") from e


class DaskSeries(dask.dataframe.core.Series, OperatorMethodMixin):
    """
    A DaskSeries object.
    """

    @deprecated("2.5.2")
    def __init__(self, series, key, transformers=[]):
        self.series = series
        self.key = key
        self.transformers = transformers

    def __getattribute__(self, item):
        return object.__getattribute__(self, "_getattribute")(item)

    def __getattr__(self, item):
        return object.__getattribute__(self, "_getattribute")(item)

    def _getattribute(self, item):
        try:
            attr = getattr(object.__getattribute__(self, "series"), item)
            if isinstance(attr, types.FunctionType) or isinstance(
                attr, types.MethodType
            ):
                return DaskSeries._apply(attr)
            elif isinstance(attr, Accessor):
                return DaskSeriesAccessor(attr, item)
            else:
                return attr
        except:
            attr = object.__getattribute__(self, item)
            return attr

    @classmethod
    def _get_unary_operator(cls, op):
        func = dask.dataframe.core.Series._get_unary_operator(op)
        return cls._apply(func)

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        func = dask.dataframe.core.Series._get_binary_operator(op, inv=inv)
        return cls._apply(func)

    @classmethod
    def _apply(cls, func, accessor_type=None, partial_func_name=None):
        def series_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, dask.dataframe.core.Series):

                def df_func(
                    df,
                    name=None,
                    func=None,
                    args=[],
                    accessor_type=None,
                    partial_func_name=None,
                    kwargs={},
                ):
                    if name is None:
                        args = [
                            df[arg] if isinstance(arg, str) else arg for arg in args
                        ]
                        return func(*args, **kwargs)
                    else:
                        if accessor_type is not None:
                            function_name = partial_func_name
                            series = getattr(df[name], accessor_type)
                        else:
                            function_name = func.__name__
                            series = df[name]

                        return getattr(series, function_name)(*args, **kwargs)

                new_kwargs = kwargs.copy()
                args = [
                    arg.name if isinstance(arg, dask.dataframe.core.Series) else arg
                    for arg in args
                ]
                new_kwargs.update(
                    {
                        "name": result.name,
                        "func": func,
                        "args": args,
                        "accessor_type": accessor_type,
                        "partial_func_name": partial_func_name,
                        "kwargs": kwargs,
                    }
                )
                ft = FunctionTransformer(
                    func=df_func, kw_args=new_kwargs, validate=False
                ).fit(result)
                result = DaskSeries(result, result.name)
                function_name = (
                    "%s-%s" % (partial_func_name, str(uuid.uuid4()))
                    if partial_func_name is not None
                    else "%s-%s" % (func.__name__, str(uuid.uuid4()))
                )
                result.transformers.append((function_name, ft))
                return result

            return result

        return series_func


# bind operators
for op in [
    operator.abs,
    operator.add,
    operator.and_,
    operator.truediv,
    operator.eq,
    operator.gt,
    operator.ge,
    operator.inv,
    operator.lt,
    operator.le,
    operator.mod,
    operator.mul,
    operator.ne,
    operator.neg,
    operator.or_,
    operator.pow,
    operator.sub,
    operator.truediv,
    operator.floordiv,
    operator.xor,
]:
    DaskSeries._bind_operator(op)


class DaskSeriesAccessor:
    """
    A DaskSeriesAccessor object.
    """

    @deprecated("2.5.2")
    def __init__(self, series_accessor, accessor_type):
        self.series_accessor = series_accessor
        self.accessor_type = accessor_type

    def __getattr__(self, item):
        attr = getattr(self.series_accessor, item)
        if (
            isinstance(attr, types.FunctionType)
            or isinstance(attr, types.MethodType)
            or isinstance(attr, partial)
        ):
            return DaskSeries._apply(
                attr, accessor_type=self.accessor_type, partial_func_name=item
            )
        else:
            return attr
