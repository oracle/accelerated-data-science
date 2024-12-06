# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import html
import json
from datetime import datetime


def parse_datetime(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def get_duration(log: dict) -> float:
    """Gets the duration of an event in seconds from a log record.
    The log record should contain two keys: `start_time` and `end_time`.
    Each of the value should be a time in string format of
    `%Y-%m-%d %H:%M:%S.%f`

    The duration is calculated by parsing two strings, and
    subtracting the `end_time` from `start_time`.

    If either `start_time` or `end_time` is not presented,
    0 will be returned.

    Parameters
    ----------
    log : dict
        A log record containing keys: `start_time` and `end_time`

    Returns
    -------
    float
        Duration in seconds.
    """
    if "end_time" not in log or "start_time" not in log:
        return 0
    return (
        parse_datetime(log.get("end_time")) - parse_datetime(log.get("start_time"))
    ).total_seconds()


def is_json_string(s):
    """Checks if a string contains valid JSON."""
    try:
        json.loads(s)
    except Exception:
        return False
    return True


def escape_html(obj):
    if isinstance(obj, dict):
        return {k: escape_html(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [escape_html(v) for v in obj]
    elif isinstance(obj, str):
        return html.escape(obj)
    return html.escape(str(obj))
