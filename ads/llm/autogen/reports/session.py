# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
"""Module for building session report."""
import copy
import json
import logging
from typing import Optional

import fsspec
import plotly.express as px
import pandas as pd
import report_creator as rc
from ads.common.auth import default_signer
from ads.llm.autogen.v02.constants import Events
from ads.llm.autogen.reports.utils import get_duration, is_json_string


logger = logging.getLogger(__name__)


class SessionReport:
    def __init__(self, log_file: str, auth: Optional[dict] = None) -> None:
        self.log_file = log_file
        if self.log_file.startswith("oci://"):
            auth = auth or default_signer()
            with fsspec.open(self.log_file, mode="r", **auth) as f:
                self.log_lines = f.readlines()
        else:
            with open(self.log_file, mode="r", encoding="utf-8") as f:
                self.log_lines = f.readlines()
        self.logs = self._parse_logs()
        self.event_logs = self.get_event_logs()
        self.invocation_logs = self._parse_invocation_events()

    @staticmethod
    def format_json_string(s):
        return f"```json\n{json.dumps(json.loads(s), indent=2)}\n```"

    def _parse_logs(self):
        logs = []
        for log in self.log_lines:
            try:
                logs.append(json.loads(log))
            except Exception as e:
                continue
        return logs

    def _parse_invocation_events(self):
        # LLM calls
        llm_events = self.filter_event_logs(Events.LLM_CALL)
        llm_call_counter = 1
        for event in llm_events:
            event["name"] = f"LLM Call {str(llm_call_counter)}"
            llm_call_counter += 1
        # Tool Calls
        tool_events = self.filter_event_logs(Events.TOOL_CALL)
        for event in tool_events:
            event["start_time"] = self.estimate_tool_call_start_time(event)
            event["name"] = event["tool_name"]
            event["end_time"] = event["timestamp"]

        events = sorted(llm_events + tool_events, key=lambda x: x.get("start_time"))
        for event in events:
            event["duration"] = get_duration(event)

        return events

    def get_event_data(self, event_name: str):
        for log in self.logs:
            if log.get(Events.KEY) == event_name:
                return log
        return None

    def filter_event_logs(self, event_name):
        filtered_logs = []
        for log in self.logs:
            if log.get(Events.KEY) == event_name:
                filtered_logs.append(log)
        return filtered_logs

    def get_event_logs(self):
        event_logs = []
        for log in self.logs:
            if Events.KEY in log:
                event_logs.append(log)
        return sorted(
            event_logs, key=lambda x: x.get("timestamp", x.get("end_time", ""))
        )

    def estimate_tool_call_start_time(self, tool_call_log):
        event_index = self.event_logs.index(tool_call_log)
        while event_index > 0:
            log = self.event_logs[event_index]

            if log.get("json_state") and (
                json.loads(log.get("json_state", "")).get("reply_func_name")
                == "check_termination_and_human_reply"
            ):
                return log.get("timestamp")
            event_index -= 1
        return None

    def build_timeline_figure(self):
        df = pd.DataFrame(self.invocation_logs)
        fig = px.timeline(
            df,
            x_start="start_time",
            x_end="end_time",
            y="name",
            color="duration",
            color_continuous_scale="rdylgn_r",
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(autorange="reversed")
        return fig

    def format_messages(self, messages):
        text = ""
        for message in messages:
            text += f"**{message.get('role')}**:\n{message.get('content')}\n\n"
        return text

    def build_llm_chat(self, llm_log):
        request = llm_log.get("request", {})
        source_name = llm_log.get("source_name")

        header = f"{source_name} invoking {request.get('model')}"

        description = f"*{llm_log.get('start_time')}*"

        request_value = f"{str(len(request.get('messages')))} messages"
        tools = request.get("tools")
        if tools:
            request_value += f", {str(len(tools))} tools"

        response = llm_log.get("response")
        response_message = response.get("choices")[0].get("message")
        response_text = response_message.get("content", "")
        tool_calls = response_message.get("tool_calls")
        if tool_calls:
            response_text += f"\n\n**Tool Calls**:"
            for tool_call in tool_calls:
                func = tool_call.get("function")
                response_text += f"\n\n`{func.get('name')}(**{func.get('arguments')})`"
        response_time = get_duration(llm_log)

        return rc.Block(
            rc.Text(
                description,
                label=header,
            ),
            rc.Group(
                rc.Block(
                    rc.Metric(
                        heading="Request",
                        value=request_value,
                        label=self.format_messages(request.get("messages")),
                    ),
                    rc.Collapse(
                        rc.Json(request),
                        label="JSON",
                    ),
                ),
                rc.Block(
                    rc.Metric(
                        heading="Response",
                        value=response_time,
                        unit="s",
                        label=response_text,
                    ),
                    rc.Collapse(
                        rc.Json(response),
                        label="JSON",
                    ),
                ),
                # label=request_header,
            ),
        )

    def build_tool_call(self, log: dict):
        source_name = log.get("source_name")
        header = f"{source_name} invoking {log.get('tool_name')}"
        request = copy.deepcopy(log)
        response = request.pop("returns", {})
        try:
            response = json.loads(response)
        except Exception:
            pass

        tool_call_args = log.get("input_args", "")
        if is_json_string(tool_call_args):
            tool_call_args = self.format_json_string(tool_call_args)

        return rc.Block(
            rc.Group(
                rc.Block(
                    rc.Metric(
                        heading="Request",
                        value=log.get("tool_name"),
                        label=tool_call_args,
                    ),
                    rc.Collapse(
                        rc.Json(request),
                        label="JSON",
                    ),
                ),
                rc.Block(
                    rc.Metric(
                        heading="Response",
                        value=get_duration(log),
                        unit="s",
                        label=str(response),
                    ),
                    rc.Collapse(
                        rc.Json(response),
                        label="JSON",
                    ),
                ),
                label=header,
            ),
        )

    def build_invocations(self, logs):
        blocks = []
        for log in logs:
            event_name = log.get(Events.KEY)
            if event_name == Events.LLM_CALL:
                blocks.append(self.build_llm_chat(log))
            elif event_name == Events.TOOL_CALL:
                blocks.append(self.build_tool_call(log))
        return blocks

    def build_chats(self):
        return rc.Block(
            rc.Group(
                rc.Block(),
                rc.Markdown("# A\nsaying something"),
            ),
            rc.Group(
                rc.Markdown("# A\nsaying something"),
                rc.Block(),
            ),
            label="Chats",
        )

    def build(self, output_file: str):
        start_event = self.get_event_data(Events.SESSION_START)
        start_time = start_event.get("timestamp")
        session_id = start_event.get("session_id")

        event_logs = self.get_event_logs()
        new_agent_logs = self.filter_event_logs(Events.NEW_AGENT)
        llm_call_logs = self.filter_event_logs(Events.LLM_CALL)
        tool_call_logs = self.filter_event_logs(Events.TOOL_CALL)

        with rc.ReportCreator(
            title=f"AutoGen Session: {session_id}",
            description=f"Started at {start_time}",
            footer="Created with ❤️ by Oracle ADS",
        ) as report:

            view = rc.Block(
                rc.Group(
                    rc.Metric(
                        heading="Agents",
                        value=len(new_agent_logs),
                    ),
                    rc.Metric(
                        heading="Events",
                        value=len(event_logs),
                    ),
                    rc.Metric(
                        heading="LLM Calls",
                        value=len(llm_call_logs),
                    ),
                    rc.Metric(
                        heading="Tool Calls",
                        value=len(tool_call_logs),
                    ),
                ),
                rc.Select(
                    blocks=[
                        rc.Widget(self.build_timeline_figure(), label="Timeline"),
                        rc.Block(
                            *self.build_invocations(self.invocation_logs),
                            label="Invocations",
                        ),
                        self.build_chats(),
                    ],
                ),
            )

            report.save(view, output_file)
