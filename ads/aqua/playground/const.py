#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.extended_enum import ExtendedEnumMeta


class Status(str, metaclass=ExtendedEnumMeta):
    """Enumeration for the status of various entities like records, sessions, etc."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class MessageRate(str, metaclass=ExtendedEnumMeta):
    """Enumeration for message rating."""

    DEFAULT = 0
    LIKE = 1
    DISLIKE = -1


class MessageRole(str, metaclass=ExtendedEnumMeta):
    """Enumeration for message roles."""

    USER = "user"
    SYSTEM = "system"


class ObjectType(str, metaclass=ExtendedEnumMeta):
    """The status of the record."""

    SESSION = "session"
    THREAD = "thread"
    MESSAGE = "message"
