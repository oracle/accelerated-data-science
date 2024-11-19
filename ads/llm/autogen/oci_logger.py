# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import io
import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

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
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import (
    CreatePreauthenticatedRequestDetails,
    PreauthenticatedRequest,
)

from ads.common.auth import default_signer
from ads.llm.autogen.constants import Events
from ads.llm.autogen.reports.session import SessionReport


logger = logging.getLogger(__name__)


def is_json_serializable(obj) -> bool:
    """Checks if an object is JSON serializable."""
    try:
        json.dumps(obj)
    except Exception:
        return False
    return True


def serialize_response(response) -> dict:
    """Serializes the LLM response to dictionary."""
    if isinstance(response, SimpleNamespace) or is_json_serializable(response):
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


class OCIFileHandler(logging.FileHandler):
    """Log handler for saving log file to OCI object storage."""

    def __init__(
        self,
        filename: str,
        session_id: str,
        mode: str = "a",
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
        auth: dict | None = None,
    ) -> None:
        self.session_id = session_id
        self.auth = auth

        if filename.startswith("oci://"):
            self.baseFilename = filename
        else:
            self.baseFilename = os.path.abspath(os.path.expanduser(filename))
            os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)

        # The following code are from the `FileHandler.__init__()`
        self.mode = mode
        self.encoding = encoding
        if "b" not in mode:
            self.encoding = io.text_encoding(encoding)
        self.errors = errors
        self.delay = delay

        if delay:
            # We don't open the stream, but we still need to call the
            # Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        auth = self.auth or default_signer()
        return fsspec.open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            **auth,
        ).open()

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
    """Represents a logging session."""

    session_id: str
    log_dir: str
    log_file: str
    thread_id: int
    pid: int
    logger: logging.Logger
    auth: dict = field(default_factory=dict)
    report_file: Optional[str] = None
    par_uri: Optional[str] = None

    def __repr__(self) -> str:
        if self.par_uri:
            return self.par_uri
        elif self.report_file:
            return self.report_file
        return self.log_file

    def create_par_uri(self, oci_file: str, **kwargs) -> str:
        """Creates a pre-authenticated request URI for a file on OCI object storage.

        Parameters
        ----------
        oci_file : str
            OCI file URI in the format of oci://bucket@namespace/prefix/to/file
        auth : dict, optional
            Dictionary containing the OCI authentication config and signer.
            Defaults to `ads.common.auth.default_signer()`.

        Returns
        -------
        str
            The pre-authenticated URI
        """
        auth = self.auth or default_signer()
        client = ObjectStorageClient(**auth)
        parsed = urlparse(oci_file)
        bucket = parsed.username
        namespace = parsed.hostname
        time_expires = kwargs.pop(
            "time_expires", datetime.now(timezone.utc) + timedelta(weeks=1)
        )
        access_type = kwargs.pop("access_type", "ObjectRead")
        response: PreauthenticatedRequest = client.create_preauthenticated_request(
            bucket_name=bucket,
            namespace_name=namespace,
            create_preauthenticated_request_details=CreatePreauthenticatedRequestDetails(
                name=os.path.basename(oci_file),
                object_name=str(parsed.path).lstrip("/"),
                access_type=access_type,
                time_expires=time_expires,
                **kwargs,
            ),
        ).data
        return response.full_path

    def create_report(self, report_file: str, return_par_uri: bool = False, **kwargs):
        auth = self.auth or default_signer()
        report = SessionReport(log_file=self.log_file, auth=auth)
        if report_file.startswith("oci://"):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the report to local temp dir
                temp_report = os.path.join(temp_dir, os.path.basename(report_file))
                report.build(temp_report)
                # Upload to OCI object storage
                fs = fsspec.filesystem("oci", **auth)
                fs.put(temp_report, report_file)
                if return_par_uri:
                    par_uri = self.create_par_uri(oci_file=report_file, **kwargs)
                    self.report_file = report_file
                    self.par_uri = par_uri
                    return par_uri
        else:
            report_file = os.path.abspath(os.path.expanduser(report_file))
            report.build(report_file)
        self.report_file = report_file
        return report_file


class OCIFileLogger(FileLogger):
    """Logger for saving log file to OCI object storage."""

    def __init__(
        self,
        log_dir: str,
        session_id: Optional[str] = None,
        auth: Optional[dict] = None,
    ):
        """Initialize a file logger for new session.

        Parameters
        ----------
        log_dir : str
            Directory for saving the log file.
        session_id : str, optional
            Session ID, by default None.
            If the session ID is None, a new UUID4 will be generated.
            The session ID will be used as the log filename.
        auth: dict, optional
            Dictionary containing the OCI authentication config and signer.
            If auth is None, `ads.common.auth.default_signer()` will be used.
        """
        self.sessions: Dict[int, LoggingSession] = {}
        self.new_session(log_dir=log_dir, session_id=session_id, auth=auth)

    @property
    def session(self) -> Optional[LoggingSession]:
        """Session for the current thread."""
        return self.sessions.get(threading.get_ident())

    @property
    def logger(self) -> Optional[str]:
        """Logger for the current thread."""
        session = self.sessions.get(threading.get_ident())
        return session.logger if session else None

    @property
    def session_id(self) -> Optional[str]:
        """Session ID for the current thread."""
        return self.sessions[threading.get_ident()].session_id

    @property
    def log_file(self) -> Optional[str]:
        """Log file path for the current session."""
        return self.sessions[threading.get_ident()].log_file

    @property
    def name(self) -> Optional[str]:
        return self.session_id or "oci_file_logger"

    def new_session(
        self,
        log_dir: str,
        session_id: Optional[str] = None,
        auth: Optional[dict] = None,
    ) -> str:
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
        auth: dict, optional
            Dictionary containing the OCI authentication config and signer.
            If auth is None, `ads.common.auth.default_signer()` will be used.


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
        log_file = os.path.join(log_dir, f"{session_id}.log")

        # Prepare the logger
        session_logger = logging.getLogger(session_id)
        session_logger.setLevel(logging.INFO)
        file_handler = OCIFileHandler(log_file, session_id=session_id, auth=auth)
        session_logger.addHandler(file_handler)

        # Create logging session
        self.sessions[thread_id] = LoggingSession(
            session_id=session_id,
            log_dir=log_dir,
            log_file=log_file,
            thread_id=thread_id,
            pid=os.getpid(),
            logger=session_logger,
            auth=auth,
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
