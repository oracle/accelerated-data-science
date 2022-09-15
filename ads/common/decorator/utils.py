#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Callable


def _get_original_func(func: callable) -> Callable:
    """The helper to retrieve the original function from the decorated one."""
    if func and hasattr(func, "__wrapped__"):
        return _get_original_func(func.__wrapped__)
    return func


class class_or_instance_method(classmethod):
    """Converts a function to be a class method or an instance depending on how it is called at runtime.

    To declare a class method, use this idiom:

        class C:
            @classmethod
            def f(obj, *args, **kwargs):
                ...
    It can be called either on the class (e.g. C.f()) or on an instance (e.g. C().f()).
    If it is called on the class C.f(), the first argument (obj) will be the class (aka. cls).
    If it is called on the instance C().f(), the first argument (obj) will be the instance (aka. self).

    """

    def __get__(self, instance, type_):
        delegate_get = super().__get__ if instance is None else self.__func__.__get__
        return delegate_get(instance, type_)
