# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
"""Module for building session report."""
import copy
import json
import logging
import os
from typing import List, Optional

import fsspec
import pandas as pd
import plotly.express as px
import report_creator as rc
from jinja2 import Environment, FileSystemLoader

from ads.common.auth import default_signer
from ads.llm.autogen.reports.utils import escape_html, get_duration, is_json_string
from ads.llm.autogen.v02.constants import Events

logger = logging.getLogger(__name__)


class SessionReport:
    def __init__(self, log_file: str, auth: Optional[dict] = None) -> None:
        self.log_file = log_file
        if self.log_file.startswith("oci://"):
            auth = auth or default_signer()
            with fsspec.open(self.log_file, mode="r", **auth) as f:
                self.log_lines = f.readlines()
        else:
            with open(self.log_file, encoding="utf-8") as f:
                self.log_lines = f.readlines()
        self.logs = self._parse_logs()
        self.event_logs = self.get_event_logs()
        self.invocation_logs = self._parse_invocation_events()
        self.received_message_logs = self._parse_received_messages()

    @staticmethod
    def format_json_string(s) -> str:
        return f"```json\n{json.dumps(json.loads(s), indent=2)}\n```"

    @staticmethod
    def _apply_template(template_path, **kwargs) -> str:
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
            return SessionReport._apply_template(
                template_path=template_path,
                sender=kwargs.get("sender", "N/A"),
                content="TEMPLATE RENDER ERROR",
                timestamp=kwargs.get("timestamp", ""),
            )
        return html

    @staticmethod
    def _preview_message(message: str, max_length=30) -> str:
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

    @staticmethod
    def _llm_call_header(log: dict):
        request = log.get("request", {})
        source_name = log.get("source_name")

        header = f"{source_name} invoking {request.get('model')}"
        if log.get("is_cached"):
            header += "(Cached)"
        return header

    @staticmethod
    def _parse_start_time(log: dict):
        start_date, start_time = log.get("start_time", " ").split(" ", 1)
        start_time = start_time.split(".", 1)[0]
        return start_date, start_time

    def _parse_logs(self) -> List[dict]:
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
            event["header"] = self._llm_call_header(event)
            llm_call_counter += 1
        # Tool Calls
        tool_events = self.filter_event_logs(Events.TOOL_CALL)
        for event in tool_events:
            event["start_time"] = self.estimate_tool_call_start_time(event)
            event["header"] = (
                f"{event.get('source_name', '')} invoking {event.get('tool_name', '')}"
            )
            event["end_time"] = event["timestamp"]

        events = sorted(llm_events + tool_events, key=lambda x: x.get("start_time"))
        for i, event in enumerate(events):
            event["header"] = f'{str(i + 1)} {event["header"]}'
            event["duration"] = get_duration(event)

        return events

    def _parse_received_messages(self):
        logs = self.filter_event_logs("new_agent")
        managers = self.get_chat_managers()
        logs = self.filter_event_logs("received_message")
        if not logs:
            return []
        states: List[dict] = [json.loads(log.get("json_state", "{}")) for log in logs]
        for i, state in enumerate(states):
            state.update(logs[i])
        logs = states
        logs = sorted(logs, key=lambda x: x.get("timestamp", ""))
        logs = [log for log in logs if log.get("sender") not in managers]
        return logs

    def get_event_data(self, event_name: str):
        for log in self.logs:
            if log.get(Events.KEY) == event_name:
                return log
        return None

    def filter_event_logs(self, event_name) -> List[dict]:
        filtered_logs = []
        for log in self.logs:
            if log.get(Events.KEY) == event_name:
                filtered_logs.append(copy.deepcopy(log))
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

    def build_timeline_tab(self):
        if not self.invocation_logs:
            return rc.Text("No LLM or Tool Calls.", label="Timeline")
        df = pd.DataFrame(self.invocation_logs)
        fig = px.timeline(
            df,
            x_start="start_time",
            x_end="end_time",
            y="header",
            labels={"header": "Invocation"},
            color="duration",
            color_continuous_scale="rdylgn_r",
            height=max(len(df.index) * 50, 500),
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(autorange="reversed")
        return rc.Block(
            rc.Widget(fig, label="Timeline"), self.build_flowchart(), label="Timeline"
        )

    def format_messages(self, messages):
        text = ""
        for message in messages:
            text += f"**{message.get('role')}**:\n{message.get('content')}\n\n"
        return text

    def build_llm_call(self, llm_log: dict):
        request = llm_log.get("request", {})
        header = llm_log.get("header", "")

        start_date, start_time = self._parse_start_time(llm_log)

        description = f" "
        if llm_log.get("is_cached"):
            description += ", **CACHED**"

        request_value = f"{str(len(request.get('messages')))} messages"
        tools = request.get("tools", [])
        if tools:
            request_value += f", {str(len(tools))} tools"

        response = llm_log.get("response")
        response_message = response.get("choices")[0].get("message")
        response_text = response_message.get("content") or ""
        tool_calls = response_message.get("tool_calls")
        if tool_calls:
            response_text += "\n\n**Tool Calls**:"
            for tool_call in tool_calls:
                func = tool_call.get("function")
                response_text += f"\n\n`{func.get('name')}(**{func.get('arguments')})`"
        duration = get_duration(llm_log)

        metrics = [
            rc.Metric(heading="Time", value=start_time, label=start_date),
            rc.Metric(
                heading="Messages",
                value=len(request.get("messages", [])),
            ),
            rc.Metric(heading="Tools", value=len(tools)),
            rc.Metric(heading="Duration", value=duration, unit="s"),
            rc.Metric(
                heading="Cached",
                value="Yes" if llm_log.get("is_cached") else "No",
            ),
            rc.Metric(heading="Cost", value=llm_log.get("cost")),
        ]

        usage = response.get("usage")
        if isinstance(usage, dict):
            for k, v in usage.items():
                if not v:
                    continue
                metrics.append(
                    rc.Metric(heading=str(k).replace("_", " ").title(), value=v)
                )

        return rc.Block(
            rc.Block(rc.Group(*metrics, label=header)),
            rc.Group(
                rc.Block(
                    rc.Markdown(
                        self.format_messages(request.get("messages")), label="Request"
                    ),
                    rc.Collapse(
                        rc.Json(request),
                        label="JSON",
                    ),
                ),
                rc.Block(
                    rc.Markdown(response_text, label="Response"),
                    rc.Collapse(
                        rc.Json(response),
                        label="JSON",
                    ),
                ),
            ),
        )

    def build_tool_call(self, log: dict):
        header = log.get("header", "")
        request = copy.deepcopy(log)
        response = request.pop("returns", {})

        start_date, start_time = self._parse_start_time(log)
        tool_call_args = log.get("input_args", "")
        if is_json_string(tool_call_args):
            tool_call_args = self.format_json_string(tool_call_args)

        if is_json_string(response):
            response = self.format_json_string(response)

        duration = get_duration(log)

        metrics = [
            rc.Metric(heading="Time", value=start_time, label=start_date),
            rc.Metric(heading="Duration", value=duration, unit="s"),
        ]

        return rc.Block(
            rc.Block(rc.Group(*metrics, label=header)),
            rc.Group(
                rc.Block(
                    rc.Markdown(
                        (log.get("tool_name") or "") + "\n\n" + tool_call_args,
                        label="Request",
                    ),
                    rc.Collapse(
                        rc.Json(request),
                        label="JSON",
                    ),
                ),
                rc.Block(rc.Text("", label="Response"), rc.Markdown(response)),
            ),
        )

    def build_invocations_tab(self) -> rc.Block:
        blocks = []
        for log in self.invocation_logs:
            event_name = log.get(Events.KEY)
            if event_name == Events.LLM_CALL:
                blocks.append(self.build_llm_call(log))
            elif event_name == Events.TOOL_CALL:
                blocks.append(self.build_tool_call(log))
        return rc.Block(
            *blocks,
            label="Invocations",
        )

    def build_flowchart(self):
        logs = self.received_message_logs
        senders = []
        for log in logs:
            sender = log.get("sender")
            senders.append(sender)

        diagram_src = "graph LR\n"
        prev_sender = None
        links = []
        # Conversation Flow
        for sender in senders:
            if prev_sender is None:
                link = f"START([START]) --> {sender}"
            else:
                link = f"{prev_sender} --> {sender}"
            if link not in links:
                links.append(link)
            prev_sender = sender
        links.append(f"{prev_sender} --> END([END])")
        # Tool Calls
        logs = self.filter_event_logs(Events.TOOL_CALL)
        for log in logs:
            tool = log.get("tool_name")
            agent = log.get("source_name")
            if tool and agent:
                link = f"{agent} <--> {tool}[[{tool}]]"
            if link not in links:
                links.append(link)

        diagram_src += "\n".join(links)
        print(diagram_src)
        return rc.Diagram(src=diagram_src, label="Flowchart")

    def build_chat_tab(self) -> rc.Block:
        logs = copy.deepcopy(self.received_message_logs)
        if not logs:
            return rc.Text("No messages received in this session.")
        # The agent sending the first message will be placed on the right.
        # All other agents will be placed on the left
        host = logs[0].get("sender")
        blocks = []

        for log in copy.deepcopy(self.received_message_logs):
            sender = log.get("sender")
            message = log.get("message")
            # Content
            if isinstance(message, dict) and "content" in message:
                content = message.get("content")
                if is_json_string(content):
                    log["json_content"] = json.dumps(json.loads(content), indent=2)
                log["content"] = content
            else:
                log["content"] = message
            # Tool call
            if isinstance(message, dict) and "tool_calls" in message:
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    tool_call_signatures = []
                    for tool_call in tool_calls:
                        func = tool_call.get("function")
                        if not func:
                            continue
                        tool_call_signatures.append(
                            f'{func.get("name")}(**{func.get("arguments", "{}")})'
                        )
                    log["tool_calls"] = tool_call_signatures
            if sender == host:
                html = self._apply_template("chat_box_rt.html", **log)
            else:
                html = self._apply_template("chat_box_lt.html", **log)
            blocks.append(rc.Html(html))

        return rc.Block(
            *blocks,
            label="Chats",
        )

    def build_logs_tab(self) -> rc.Block:
        blocks = []
        for log_line in self.log_lines:
            if is_json_string(log_line):
                log = json.loads(log_line)
                label = log.get(
                    "event_name", self._preview_message(log.get("message", ""))
                )
                blocks.append(rc.Collapse(rc.Json(escape_html(log)), label=label))
            else:
                log = log_line
                blocks.append(
                    rc.Collapse(rc.Text(log), label=self._preview_message(log_line))
                )

        return rc.Block(
            *blocks,
            label="Logs",
        )

    def count_agents(self):
        # AutoGen may have new_agent multiple times
        # Here we count the number agents by name
        new_agent_logs = self.filter_event_logs(Events.NEW_AGENT)
        agents = set()
        for log in new_agent_logs:
            agents.add((log.get("agent_name"), log.get("agent_type")))
        return len(agents)

    def get_chat_managers(self):
        # AutoGen may have new_agent multiple times
        # Here we count the number agents by name
        new_agent_logs = self.filter_event_logs(Events.NEW_AGENT)
        agents = set()
        for log in new_agent_logs:
            if not log.get("is_manager"):
                continue
            agents.add(log.get("agent_name"))
        return agents

    def build(self, output_file: str):
        start_event = self.get_event_data(Events.SESSION_START)
        start_time = start_event.get("timestamp")
        session_id = start_event.get("session_id")

        event_logs = self.get_event_logs()

        llm_call_logs = self.filter_event_logs(Events.LLM_CALL)
        tool_call_logs = self.filter_event_logs(Events.TOOL_CALL)

        chat_managers = self.get_chat_managers()
        if not chat_managers:
            agent_label = ""
        elif len(chat_managers) == 1:
            agent_label = "+1 chat manager"
        else:
            agent_label = f"+{str(len(chat_managers))} chat managers"

        with rc.ReportCreator(
            title=f"AutoGen Session: {session_id}",
            description=f"Started at {start_time}",
            footer="Created with ❤️ by Oracle ADS",
        ) as report:

            view = rc.Block(
                rc.Group(
                    rc.Metric(
                        heading="Agents",
                        value=self.count_agents() - len(chat_managers),
                        label=agent_label,
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
                        self.build_timeline_tab(),
                        self.build_invocations_tab(),
                        self.build_chat_tab(),
                        self.build_logs_tab(),
                    ],
                ),
            )

            report.save(view, output_file)
