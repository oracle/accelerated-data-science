#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import Dict, List

from jinja2 import Environment, PackageLoader

from ads.aqua.app import AquaApp
from ads.function.service.oci_function import OCIFunction


class AquaToolApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua Tools."""

    def get(self, id: str) -> Dict:
        """Gets the information of an Aqua tool."""
        return OCIFunction.from_id(id)

    def manual(self, id: str) -> str:
        """Gets the information of an Aqua tool."""
        from rich.console import Console
        from rich.markdown import Markdown

        oci_function = OCIFunction.from_id(id)

        env = Environment(loader=PackageLoader("ads", "aqua/tool/templates"))
        score_template = env.get_template("manual.md.jinja2")

        context = {
            "tool_name": oci_function.display_name,
            "tool_ocid": oci_function.id,
            "time_created": oci_function.time_created,
            "tool_openapi_spec": json.dumps(
                json.loads(oci_function.freeform_tags.get("OPENAPI_SCHEMA")), indent=2
            ),
            "tool_payload_example": oci_function.freeform_tags.get("PAYLOAD_EXAMPLE"),
        }

        Console().print(Markdown(score_template.render(context)))
        return ""

    def list(self, application_id: str = None) -> List[Dict]:
        """List Aqua tools in a given compartment."""
        return OCIFunction().list(application_id=application_id)

    def invoke(self, id: str, payload: Dict):
        """Runs the tool."""

        return OCIFunction.from_id(id).invoke(payload)

    def watch(self, id: str, interval: int = 3):
        """Watches the logs."""
        OCIFunction.from_id(id).watch(interval=interval)
