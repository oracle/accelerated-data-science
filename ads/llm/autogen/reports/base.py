# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
import logging
import os

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class BaseReport:
    """Base class containing utilities for generating reports."""

    @staticmethod
    def format_json_string(s) -> str:
        """Formats the JSON string in markdown."""
        return f"```json\n{json.dumps(json.loads(s), indent=2)}\n```"

    @staticmethod
    def _parse_date_time(datetime_string: str):
        """Parses a datetime string in the logs into date and time.
        Keeps only the seconds in the time.
        """
        date_str, time_str = datetime_string.split(" ", 1)
        time_str = time_str.split(".", 1)[0]
        return date_str, time_str

    @staticmethod
    def _preview_message(message: str, max_length=30) -> str:
        """Shows the beginning part of a string message."""
        # Return the entire string if it is less than the max_length
        if len(message) <= max_length:
            return message
        # Go backward until we find the first whitespace
        idx = 30
        while not message[idx].isspace() and idx > 0:
            idx -= 1
        # If we found a whitespace
        if idx > 0:
            return message[:idx] + "..."
        # If we didn't find a whitespace
        return message[:30] + "..."

    @classmethod
    def _render_template(cls, template_path, **kwargs) -> str:
        """Render Jinja template with kwargs."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        environment = Environment(
            loader=FileSystemLoader(template_dir), autoescape=True
        )
        template = environment.get_template(template_path)
        try:
            html = template.render(**kwargs)
        except Exception:
            logger.error(
                "Unable to render template %s with data:\n%s",
                template_path,
                str(kwargs),
            )
            return cls._render_template(
                template_path=template_path,
                sender=kwargs.get("sender", "N/A"),
                content="TEMPLATE RENDER ERROR",
                timestamp=kwargs.get("timestamp", ""),
            )
        return html
