# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Module for building session report."""
import copy
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

import fsspec
import pandas as pd
import plotly.express as px
import report_creator as rc

from ads.common.auth import default_signer
from ads.llm.autogen.constants import Events
from ads.llm.autogen.reports.base import BaseReport
from ads.llm.autogen.reports.data import (
    AgentData,
    LLMCompletionData,
    LogRecord,
    ToolCallData,
)
from ads.llm.autogen.reports.utils import escape_html, get_duration, is_json_string

logger = logging.getLogger(__name__)


@dataclass
class AgentInvocation:
    """Represents an agent invocation."""

    log: LogRecord
    header: str = ""
    description: str = ""
    duration: Optional[float] = None


class SessionReport(BaseReport):
    """Class for building session report from session log file."""

    def __init__(self, log_file: str, auth: Optional[dict] = None) -> None:
        """Initialize the session report with log file.
        It is assumed that the file contains logs for a single session.

        Parameters
        ----------
        log_file : str
            Path or URI of the log file.
        auth : dict, optional
            Authentication signer/config for OCI, by default None
        """
        self.log_file: str = log_file
        if self.log_file.startswith("oci://"):
            auth = auth or default_signer()
            with fsspec.open(self.log_file, mode="r", **auth) as f:
                self.log_lines = f.readlines()
        else:
            with open(self.log_file, encoding="utf-8") as f:
                self.log_lines = f.readlines()
        self.logs: List[LogRecord] = self._parse_logs()

        # Parse logs to get entities for building the report
        # Agents
        self.agents: List[AgentData] = self._parse_agents()
        self.managers: List[AgentData] = self._parse_managers()
        # Events
        self.start_event: LogRecord = self._parse_start_event()
        self.session_id: str = self.start_event.session_id
        self.llm_calls: List[AgentInvocation] = self._parse_llm_calls()
        self.tool_calls: List[AgentInvocation] = self._parse_tool_calls()
        self.invocations: List[AgentInvocation] = self._parse_invocations()

        self.received_message_logs = self._parse_received_messages()

    def _parse_logs(self) -> List[LogRecord]:
        """Parses the logs form strings into LogRecord objects."""
        logs = []
        for i, log in enumerate(self.log_lines):
            try:
                logs.append(LogRecord.from_dict(json.loads(log)))
            except Exception as e:
                logger.error(
                    "Error when parsing log record at line %s:\n%s", str(i + 1), str(e)
                )
                continue
        # Sort the logs by timestamp
        logs = sorted(logs, key=lambda x: x.timestamp)
        return logs

    def _parse_agents(self) -> List[AgentData]:
        """Parses the logs to identify unique agents.
        AutoGen may have new_agent multiple times.
        Here we identify the agents by the unique tuple of (name, module, class).
        """
        new_agent_logs = self.filter_by_event(Events.NEW_AGENT)
        agents = {}
        for log in new_agent_logs:
            agent: AgentData = log.data
            agents[(agent.agent_name, agent.agent_module, agent.agent_class)] = agent
        return list(agents.values())

    def _parse_managers(self) -> List[AgentData]:
        """Parses the logs to get chat managers."""
        managers = []
        for agent in self.agents:
            if agent.is_manager:
                managers.append(agent)
        return managers

    def _parse_start_event(self) -> LogRecord:
        """Parses the logs to get the first logging_session_start event log."""
        records = self.filter_by_event(event_name=Events.SESSION_START)
        if not records:
            raise ValueError("logging_session_start event is not found in the logs.")
        records = sorted(records, key=lambda x: x.timestamp)
        return records[0]

    def _parse_llm_calls(self) -> List[AgentInvocation]:
        """Parses the logs to get the LLM calls."""
        records = self.filter_by_event(Events.LLM_CALL)
        invocations = []
        for record in records:
            log_data: LLMCompletionData = record.data
            source_name = record.source_name
            request = log_data.request
            # If there is no request, the log is invalid.
            if not request:
                continue

            header = f"{source_name} invoking {request.get('model')}"
            if log_data.is_cached:
                header += " (Cached)"
            invocations.append(
                AgentInvocation(
                    header=header,
                    log=record,
                    duration=get_duration(log_data.start_time, log_data.end_time),
                )
            )
        return invocations

    def _parse_tool_calls(self) -> List[AgentInvocation]:
        """Parses the logs to get the tool calls."""
        records = self.filter_by_event(Events.TOOL_CALL)
        invocations = []
        for record in records:
            log_data: ToolCallData = record.data
            source_name = record.source_name
            invocations.append(
                AgentInvocation(
                    log=record,
                    header=f"{source_name} invoking {log_data.tool_name}",
                    duration=get_duration(log_data.start_time, log_data.end_time),
                )
            )
        return invocations

    def _parse_invocations(self) -> List[AgentInvocation]:
        """Add numbering to the combined list of LLM and tool calls."""
        invocations = self.llm_calls + self.tool_calls
        invocations = sorted(invocations, key=lambda x: x.log.data.start_time)
        for i, invocation in enumerate(invocations):
            invocation.header = f"{str(i + 1)} {invocation.header}"
        return invocations

    def _parse_received_messages(self) -> List[LogRecord]:
        """Parses the logs to get the received_message events."""
        managers = [manager.agent_name for manager in self.managers]
        logs = self.filter_by_event(Events.RECEIVED_MESSAGE)
        if not logs:
            return []
        logs = sorted(logs, key=lambda x: x.timestamp)
        logs = [log for log in logs if log.kwargs.get("sender") not in managers]
        return logs

    def filter_by_event(self, event_name: str) -> List[LogRecord]:
        """Filters the logs by event name.

        Parameters
        ----------
        event_name : str
            Name of the event.

        Returns
        -------
        List[LogRecord]
            A list of LogRecord objects for the event.
        """
        filtered_logs = []
        for log in self.logs:
            if log.event_name == event_name:
                filtered_logs.append(log)
        return filtered_logs

    def _build_flowchart(self):
        """Builds the flowchart of agent chats."""
        senders = []
        for log in self.received_message_logs:
            sender = log.kwargs.get("sender")
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
        for invocation in self.tool_calls:
            tool = invocation.log.data.tool_name
            agent = invocation.log.data.agent_name
            if tool and agent:
                link = f"{agent} <--> {tool}[[{tool}]]"
            if link not in links:
                links.append(link)

        diagram_src += "\n".join(links)
        return rc.Diagram(src=diagram_src, label="Flowchart")

    def _build_timeline_tab(self):
        """Builds the plotly timeline chart."""
        if not self.invocations:
            return rc.Text("No LLM or Tool Calls.", label="Timeline")
        invocations = []
        for invocation in self.invocations:
            invocations.append(
                {
                    "start_time": invocation.log.data.start_time,
                    "end_time": invocation.log.data.end_time,
                    "header": invocation.header,
                    "duration": invocation.duration,
                }
            )
        df = pd.DataFrame(invocations)
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
            rc.Widget(fig, label="Timeline"), self._build_flowchart(), label="Timeline"
        )

    def _format_messages(self, messages: List[dict]):
        """Formats the LLM call messages to be displayed in the report."""
        text = ""
        for message in messages:
            text += f"**{message.get('role')}**:\n{message.get('content')}\n\n"
        return text

    def _build_llm_call(self, invocation: AgentInvocation):
        """Builds the LLM call details."""
        log_data: LLMCompletionData = invocation.log.data
        request = log_data.request
        response = log_data.response

        start_date, start_time = self._parse_date_time(log_data.start_time)

        request_value = f"{str(len(request.get('messages')))} messages"
        tools = request.get("tools", [])
        if tools:
            request_value += f", {str(len(tools))} tools"

        response_message = response.get("choices")[0].get("message")
        response_text = response_message.get("content") or ""
        tool_calls = response_message.get("tool_calls")
        if tool_calls:
            response_text += "\n\n**Tool Calls**:"
            for tool_call in tool_calls:
                func = tool_call.get("function")
                response_text += f"\n\n`{func.get('name')}(**{func.get('arguments')})`"

        metrics = [
            rc.Metric(heading="Time", value=start_time, label=start_date),
            rc.Metric(
                heading="Messages",
                value=len(request.get("messages", [])),
            ),
            rc.Metric(heading="Tools", value=len(tools)),
            rc.Metric(heading="Duration", value=invocation.duration, unit="s"),
            rc.Metric(
                heading="Cached",
                value="Yes" if log_data.is_cached else "No",
            ),
            rc.Metric(heading="Cost", value=log_data.cost),
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
            rc.Block(rc.Group(*metrics, label=invocation.header)),
            rc.Group(
                rc.Block(
                    rc.Markdown(
                        self._format_messages(request.get("messages")), label="Request"
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

    def _build_tool_call(self, invocation: AgentInvocation):
        """Builds the tool call details."""
        log_data: ToolCallData = invocation.log.data
        request = log_data.to_dict()
        response = request.pop("returns", {})

        start_date, start_time = self._parse_date_time(log_data.start_time)
        tool_call_args = log_data.input_args
        if is_json_string(tool_call_args):
            tool_call_args = self.format_json_string(tool_call_args)

        if is_json_string(response):
            response = self.format_json_string(response)

        metrics = [
            rc.Metric(heading="Time", value=start_time, label=start_date),
            rc.Metric(heading="Duration", value=invocation.duration, unit="s"),
        ]

        return rc.Block(
            rc.Block(rc.Group(*metrics, label=invocation.header)),
            rc.Group(
                rc.Block(
                    rc.Markdown(
                        (log_data.tool_name or "") + "\n\n" + tool_call_args,
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

    def _build_invocations_tab(self) -> rc.Block:
        """Builds the invocations tab."""
        blocks = []
        for invocation in self.invocations:
            event_name = invocation.log.event_name
            if event_name == Events.LLM_CALL:
                blocks.append(self._build_llm_call(invocation))
            elif event_name == Events.TOOL_CALL:
                blocks.append(self._build_tool_call(invocation))
        return rc.Block(
            *blocks,
            label="Invocations",
        )

    def _build_chat_tab(self) -> rc.Block:
        """Builds the chat tab."""
        if not self.received_message_logs:
            return rc.Text("No messages received in this session.", label="Chats")
        # The agent sending the first message will be placed on the right.
        # All other agents will be placed on the left
        host = self.received_message_logs[0].kwargs.get("sender")
        blocks = []

        for log in self.received_message_logs:
            context = copy.deepcopy(log.kwargs)
            context.update(log.to_dict())
            sender = context.get("sender")
            message = context.get("message", "")
            # Content
            if isinstance(message, dict) and "content" in message:
                content = message.get("content", "")
                if is_json_string(content):
                    context["json_content"] = json.dumps(json.loads(content), indent=2)
                context["content"] = content
            else:
                context["content"] = message
            if context["content"] is None:
                context["content"] = ""
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
                    context["tool_calls"] = tool_call_signatures
            if sender == host:
                html = self._render_template("chat_box_rt.html", **context)
            else:
                html = self._render_template("chat_box_lt.html", **context)
            blocks.append(rc.Html(html))

        return rc.Block(
            *blocks,
            label="Chats",
        )

    def _build_logs_tab(self) -> rc.Block:
        """Builds the logs tab."""
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

    def _build_errors_tab(self) -> Optional[rc.Block]:
        """Builds the error tab to show exception."""
        errors = self.filter_by_event(Events.EXCEPTION)
        if not errors:
            return None
        blocks = []
        for error in errors:
            label = f'{error.kwargs.get("exc_type", "")} - {error.kwargs.get("exc_value", "")}'
            variables: dict = error.kwargs.get("locals", {})
            table = "| Variable | Value |\n|---|---|\n"
            table += "\n".join([f"| {k} | {v} |" for k, v in variables.items()])
            blocks += [
                rc.Unformatted(text=error.kwargs.get("traceback", ""), label=label),
                rc.Markdown(table),
            ]
        return rc.Block(*blocks, label="Error")

    def build(self, output_file: str):
        """Builds the session report.

        Parameters
        ----------
        output_file : str
            Local path or OCI object storage URI to save the report HTML file.
        """

        if not self.managers:
            agent_label = ""
        elif len(self.managers) == 1:
            agent_label = "+1 chat manager"
        else:
            agent_label = f"+{str(len(self.managers))} chat managers"

        blocks = [
            self._build_timeline_tab(),
            self._build_invocations_tab(),
            self._build_chat_tab(),
            self._build_logs_tab(),
        ]

        error_block = self._build_errors_tab()
        if error_block:
            blocks.append(error_block)

        with rc.ReportCreator(
            title=f"AutoGen Session: {self.session_id}",
            description=f"Started at {self.start_event.timestamp}",
            footer="Created with ❤️ by Oracle ADS",
        ) as report:

            view = rc.Block(
                rc.Group(
                    rc.Metric(
                        heading="Agents",
                        value=len(self.agents) - len(self.managers),
                        label=agent_label,
                    ),
                    rc.Metric(
                        heading="Events",
                        value=len(self.logs),
                    ),
                    rc.Metric(
                        heading="LLM Calls",
                        value=len(self.llm_calls),
                    ),
                    rc.Metric(
                        heading="Tool Calls",
                        value=len(self.tool_calls),
                    ),
                ),
                rc.Select(blocks=blocks),
            )

            report.save(view, output_file)
