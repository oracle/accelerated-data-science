#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Contains the data structure for logging and reporting."""
import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from ads.llm.autogen.constants import Events


@dataclass
class LogData:
    """Base class for the data field of LogRecord."""

    def to_dict(self):
        """Convert the log data to dictionary."""
        return asdict(self)


@dataclass
class LogRecord:
    """Represents a log record.

    The `data` field is for pre-defined structured data, which should be an instance of LogData.
    The `kwargs` field is for freeform key value pairs.
    """

    session_id: str
    thread_id: int
    timestamp: str
    event_name: str
    source_id: Optional[int] = None
    source_name: Optional[str] = None
    # Structured data for specific type of logs
    data: Optional[LogData] = None
    # Freeform data
    kwargs: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert the log record to dictionary."""
        return asdict(self)

    def to_string(self):
        """Serialize the log record to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "LogRecord":
        """Initializes a LogRecord object from dictionary."""
        event_mapping = {
            Events.NEW_AGENT: AgentData,
            Events.TOOL_CALL: ToolCallData,
            Events.LLM_CALL: LLMCompletionData,
        }
        if Events.KEY not in data:
            raise KeyError("event_name not found in data.")

        data = copy.deepcopy(data)

        event_name = data["event_name"]
        if event_name in event_mapping and data.get("data"):
            data["data"] = event_mapping[event_name](**data.pop("data"))

        return cls(**data)


@dataclass
class AgentData(LogData):
    """Represents agent log Data."""

    agent_name: str
    agent_class: str
    agent_module: Optional[str] = None
    is_manager: Optional[bool] = None


@dataclass
class LLMCompletionData(LogData):
    """Represents LLM completion log data."""

    invocation_id: str
    request: dict
    response: dict
    start_time: str
    end_time: str
    cost: Optional[float] = None
    is_cached: Optional[bool] = None


@dataclass
class ToolCallData(LogData):
    """Represents tool call log data."""

    tool_name: str
    start_time: str
    end_time: str
    agent_name: str
    agent_class: str
    agent_module: Optional[str] = None
    input_args: dict = field(default_factory=dict)
    returns: Optional[Union[str, list, dict, tuple]] = None
