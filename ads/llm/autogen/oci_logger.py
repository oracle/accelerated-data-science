# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import fsspec
from autogen import Agent, ConversableAgent
from autogen.logger.file_logger import (
    ChatCompletion,
    F,
    FileLogger,
    get_current_ts,
    safe_serialize,
    to_dict,
)


logger = logging.getLogger(__name__)


def is_json_serializable(obj):
    """Checks if an object is JSON serializable."""
    try:
        json.dumps(obj)
    except Exception:
        return False
    return True


def serialize_response(response):
    if is_json_serializable(response):
        # No need to do anything
        return response
    elif isinstance(response, SimpleNamespace):
        # Convert simpleNamespace to dict
        return json.loads(json.dumps(response, default=vars))
    elif hasattr(response, "dict") and callable(response.dict):
        return response.dict()
    data = {
        "model": response.model,
        "choices": [
            {"message": {"content": choice.message.content}}
            for choice in response.choices
        ],
        "response": str(response),
    }
    return data


class Events:
    KEY = "event_name"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    NEW_AGENT = "new_agent"
    SESSION_START = "logging_session_start"
    SESSION_STOP = "logging_session_stop"


class OCIFileHandler(logging.FileHandler):
    def __init__(
        self,
        filename: str | os.PathLike[str],
        session_id: str,
        mode: str = "a",
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        super().__init__(filename, mode, encoding, delay, errors)
        self.session_id = session_id

    def format(self, record: logging.LogRecord):
        """Formats the log record as JSON payload and add session_id."""
        msg = record.getMessage()
        try:
            data = json.loads(msg)
        except Exception as e:
            data = {"message": msg}

        if "session_id" not in data:
            data["session_id"] = self.session_id
        if "thread_id" not in data:
            data["thread_id"] = threading.get_ident()

        record.msg = json.dumps(data)
        return super().format(record)


@dataclass
class LoggingSession:
    session_id: str
    log_dir: str
    log_file: str
    thread_id: int
    pid: int
    logger: logging.Logger


class OCIFileLogger(FileLogger):
    def __init__(self, log_dir: str, session_id: Optional[str] = None):
        self.sessions: Dict[int, LoggingSession] = {}
        self.new_session(log_dir=log_dir, session_id=session_id)

    @property
    def session(self):
        """Session for the current thread."""
        return self.sessions.get(threading.get_ident())

    @property
    def logger(self):
        """Logger for the current thread."""
        session = self.sessions.get(threading.get_ident())
        return session.logger if session else None

    @property
    def session_id(self):
        """Session ID for the current thread."""
        return self.sessions[threading.get_ident()].session_id

    @property
    def log_file(self):
        """Log file for the current session."""
        return self.sessions[threading.get_ident()].log_file

    @property
    def name(self):
        return self.session_id or "oci_file_logger"

    def new_session(self, log_dir: str, session_id: Optional[str] = None):
        """Creates a new logging session.

        If an active logging session is already started in the thread, the existing session will be used.

        Parameters
        ----------
        log_dir : str
            Directory for saving the log file.
        session_id : str, optional
            Session ID, by default None.
            If the session ID is None, a new UUID4 will be generated.
            The session ID will be used as the log filename.

        Returns
        -------
        str
            session ID
        """
        thread_id = threading.get_ident()

        if thread_id in self.sessions:
            logger.warning(
                "An active logging session (ID=%s) is already started in this thread (%s). "
                "Please stop the active session before starting a new session.",
                self.session_id,
                thread_id,
            )
            return self.session_id

        session_id = session_id or str(uuid.uuid4())
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        log_file = os.path.join(log_dir, f"{session_id}.log")

        # Test opening the log file
        os.makedirs(log_dir, exist_ok=True)
        try:
            with open(log_file, "a"):
                pass
        except Exception as e:
            logger.error(f"Failed to write logging file: {e}")

        # Prepare the logger
        session_logger = logging.getLogger(session_id)
        session_logger.setLevel(logging.INFO)
        file_handler = OCIFileHandler(log_file, session_id=session_id)
        session_logger.addHandler(file_handler)

        # Create logging session
        self.sessions[thread_id] = LoggingSession(
            session_id=session_id,
            log_dir=log_dir,
            log_file=log_file,
            thread_id=thread_id,
            pid=os.getpid(),
            logger=session_logger,
        )

        logger.info("Start logging session %s to file %s", session_id, log_file)
        return session_id

    def start(self) -> str:
        """Start the logging session and return the session_id."""
        self.log_event(source=self, name=Events.SESSION_START)
        return self.session_id

    def stop(self) -> None:
        """Stops the logging session."""
        self.log_event(source=self, name=Events.SESSION_STOP)
        return super().stop()

    def log_chat_completion(
        self,
        invocation_id: uuid.UUID,
        client_id: int,
        wrapper_id: int,
        source: Union[str, Agent],
        request: Dict[str, Union[float, str, List[Dict[str, str]]]],
        response: Union[str, ChatCompletion],
        is_cached: int,
        cost: float,
        start_time: str,
    ) -> None:
        """
        Log a chat completion.
        """
        thread_id = threading.get_ident()
        source_name = None
        if isinstance(source, str):
            source_name = source
        else:
            source_name = source.name

        try:
            log_data = json.dumps(
                {
                    Events.KEY: Events.LLM_CALL,
                    "invocation_id": str(invocation_id),
                    "client_id": client_id,
                    "wrapper_id": wrapper_id,
                    "request": to_dict(request),
                    "response": serialize_response(response),
                    "is_cached": is_cached,
                    "cost": cost,
                    "start_time": start_time,
                    "end_time": get_current_ts(),
                    "thread_id": thread_id,
                    "source_name": source_name,
                }
            )

            self.logger.info(log_data)
        except Exception as e:
            self.logger.error(f"[file_logger] Failed to log chat completion: {e}")

    def log_function_use(
        self, source: Union[str, Agent], function: F, args: Dict[str, Any], returns: Any
    ) -> None:
        """
        Log a registered function(can be a tool) use from an agent or a string source.
        """
        thread_id = threading.get_ident()

        try:
            log_data = json.dumps(
                {
                    Events.KEY: Events.TOOL_CALL,
                    "source_id": id(source),
                    "source_name": (
                        str(source.name) if hasattr(source, "name") else source
                    ),
                    "agent_module": source.__module__,
                    "agent_class": source.__class__.__name__,
                    "tool_name": function.__name__,
                    # This is the tool call end time
                    "timestamp": get_current_ts(),
                    "thread_id": thread_id,
                    "input_args": safe_serialize(args),
                    "returns": safe_serialize(returns),
                }
            )
            self.logger.info(log_data)
        except Exception as e:
            self.logger.error(f"[file_logger] Failed to log event {e}")

    def log_new_agent(
        self, agent: ConversableAgent, init_args: Dict[str, Any] = {}
    ) -> None:
        """
        Log a new agent instance.
        """
        thread_id = threading.get_ident()

        try:
            log_data = json.dumps(
                {
                    Events.KEY: Events.NEW_AGENT,
                    "id": id(agent),
                    "agent_name": (
                        agent.name
                        if hasattr(agent, "name") and agent.name is not None
                        else ""
                    ),
                    "wrapper_id": to_dict(
                        agent.client.wrapper_id
                        if hasattr(agent, "client") and agent.client is not None
                        else ""
                    ),
                    "session_id": self.session_id,
                    "current_time": get_current_ts(),
                    "agent_type": type(agent).__name__,
                    "args": to_dict(init_args),
                    "thread_id": thread_id,
                }
            )
            self.logger.info(log_data)
        except Exception as e:
            self.logger.error(f"[file_logger] Failed to log new agent: {e}")
