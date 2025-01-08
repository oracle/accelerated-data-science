# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import html
import json
from datetime import datetime


def parse_datetime(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def get_duration(start_time: str, end_time: str) -> float:
    """Gets the duration in seconds between `start_time` and `end_time`.
    Each of the value should be a time in string format of
    `%Y-%m-%d %H:%M:%S.%f`

    The duration is calculated by parsing the two strings,
    then subtracting the `end_time` from `start_time`.

    If either `start_time` or `end_time` is not presented,
    0 will be returned.

    Parameters
    ----------
    start_time : str
        The start time.
    end_time : str
        The end time.

    Returns
    -------
    float
        Duration in seconds.
    """
    if not start_time or not end_time:
        return 0
    return (parse_datetime(end_time) - parse_datetime(start_time)).total_seconds()


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
