# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import importlib
import inspect
import json
import logging
import os
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import autogen
import fsspec
import oci
from autogen import Agent, ConversableAgent, GroupChatManager
from autogen.logger.file_logger import (
    ChatCompletion,
    F,
    FileLogger,
    get_current_ts,
    safe_serialize,
)
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import (
    CreatePreauthenticatedRequestDetails,
    PreauthenticatedRequest,
)

import ads
from ads.common.auth import default_signer
from ads.llm.autogen.reports.session import SessionReport
from ads.llm.autogen.v02.constants import Events
from ads.llm.autogen.v02.log_handlers.oci_file_handler import OCIFileHandler

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


def serialize(
    obj: Union[int, float, str, bool, Dict[Any, Any], List[Any], Tuple[Any, ...], Any],
    exclude: Tuple[str, ...] = (),
    no_recursive: Tuple[Any, ...] = (),
) -> Any:
    try:
        if isinstance(obj, (int, float, str, bool)):
            return obj
        elif callable(obj):
            return inspect.getsource(obj).strip()
        elif isinstance(obj, dict):
            return {
                str(k): (
                    serialize(str(v))
                    if isinstance(v, no_recursive)
                    else serialize(v, exclude, no_recursive)
                )
                for k, v in obj.items()
                if k not in exclude
            }
        elif isinstance(obj, (list, tuple)):
            return [
                (
                    serialize(str(v))
                    if isinstance(v, no_recursive)
                    else serialize(v, exclude, no_recursive)
                )
                for v in obj
            ]
        else:
            return str(obj)
    except Exception:
        return str(obj)


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

    @property
    def report(self) -> str:
        if self.par_uri:
            return self.par_uri
        elif self.report_file:
            return self.report_file
        return None

    def __repr__(self) -> str:
        if self.report:
            return self.report
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

    def create_report(
        self, report_file: str, return_par_uri: bool = False, **kwargs
    ) -> str:
        """Creates a report in HTML format.

        Parameters
        ----------
        report_file : str
            The file path to save the report.
        return_par_uri : bool, optional
            If the report is saved in object storage,
            whether to create a pre-authenticated link for the report, by default False.
            This will be ignored if the report is not saved in object storage.

        Returns
        -------
        str
            The full path or pre-authenticated link of the report.
        """
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
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            report.build(report_file)
        self.report_file = report_file
        return report_file


class SessionLogger(FileLogger):
    """Logger for saving log file to OCI object storage."""

    def __init__(
        self,
        log_dir: str,
        report_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        auth: Optional[dict] = None,
        log_for_all_threads: str = False,
        report_par_uri: bool = False,
        par_kwargs: Optional[dict] = None,
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
        log_for_all_threads:
            Indicate if the logger should handle logging for all threads.
            Defaults to False, the logger will only log for the current thread.
        """
        self.report_dir = report_dir
        self.report_par_uri = report_par_uri
        self.par_kwargs = par_kwargs
        self.log_for_all_threads = log_for_all_threads

        self.session = self.new_session(
            log_dir=log_dir, session_id=session_id, auth=auth
        )

    @property
    def logger(self) -> Optional[str]:
        """Logger for the thread."""
        thread_id = threading.get_ident()
        if not self.log_for_all_threads and thread_id != self.session.thread_id:
            return None
        return self.session.logger

    @property
    def session_id(self) -> Optional[str]:
        """Session ID for the current session."""
        return self.session.session_id

    @property
    def log_file(self) -> Optional[str]:
        """Log file path for the current session."""
        return self.session.log_file

    @property
    def name(self) -> str:
        """Name of the logger."""
        return self.session_id or "oci_file_logger"

    def new_session(
        self,
        log_dir: str,
        session_id: Optional[str] = None,
        auth: Optional[dict] = None,
    ) -> LoggingSession:
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
        LoggingSession
            The new logging session
        """
        thread_id = threading.get_ident()

        session_id = str(session_id or uuid.uuid4())
        log_file = os.path.join(log_dir, f"{session_id}.log")

        # Prepare the logger
        session_logger = logging.getLogger(session_id)
        session_logger.setLevel(logging.INFO)
        file_handler = OCIFileHandler(log_file, session_id=session_id, auth=auth)
        session_logger.addHandler(file_handler)

        # Create logging session
        session = LoggingSession(
            session_id=session_id,
            log_dir=log_dir,
            log_file=log_file,
            thread_id=thread_id,
            pid=os.getpid(),
            logger=session_logger,
            auth=auth,
        )

        logger.info("Start logging session %s to file %s", session_id, log_file)
        return session

    def generate_report(
        self,
        report_dir: Optional[str] = None,
        report_par_uri: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Generates a report for the session.

        Parameters
        ----------
        report_dir : str, optional
            Directory for saving the report, by default None
        report_par_uri : bool, optional
            Whether to create a pre-authenticated link for the report, by default None.
            If the `report_par_uri` is not set, the value of `self.report_par_uri` will be used.

        Returns
        -------
        str
            The link to the report.
            If the `report_dir` is local, the local file path will be returned.
            If a pre-authenticated link is created, the link will be returned.
        """
        report_dir = report_dir or self.report_dir
        report_par_uri = (
            report_par_uri if report_par_uri is not None else self.report_par_uri
        )
        kwargs = kwargs or self.par_kwargs or {}

        report_file = os.path.join(self.report_dir, f"{self.session_id}.html")
        return self.session.create_report(
            report_file=report_file, return_par_uri=self.report_par_uri, **kwargs
        )

    def start(self) -> str:
        """Start the logging session and return the session_id."""
        envs = {
            "oracle-ads": ads.__version__,
            "oci": oci.__version__,
            "autogen": autogen.__version__,
        }
        libraries = [
            "langchain",
            "langchain-core",
            "langchain-community",
            "langchain-openai",
            "openai",
        ]
        for library in libraries:
            try:
                imported_library = importlib.import_module(library)
                version = imported_library.__version__
                envs[library] = version
            except Exception:
                pass
        self.log_event(source=self, name=Events.SESSION_START, environment=envs)
        return self.session_id

    def stop(self) -> None:
        """Stops the logging session."""
        self.log_event(source=self, name=Events.SESSION_STOP)
        super().stop()
        if self.report_dir:
            try:
                return self.generate_report()
            except Exception as e:
                logger.error(
                    "Failed to create session report for AutoGen session %s\n%s",
                    self.session_id,
                    str(e),
                )
                logger.debug(traceback.format_exc())

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
        if not self.logger:
            return

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
                    "request": serialize(request),
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
        if not self.logger:
            return

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
        if not self.logger:
            return

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
                    "wrapper_id": serialize(
                        agent.client.wrapper_id
                        if hasattr(agent, "client") and agent.client is not None
                        else ""
                    ),
                    "session_id": self.session_id,
                    "current_time": get_current_ts(),
                    "agent_type": type(agent).__name__,
                    "args": serialize(init_args),
                    "thread_id": thread_id,
                    "is_manager": isinstance(agent, GroupChatManager),
                }
            )
            self.logger.info(log_data)
        except Exception as e:
            self.logger.error(f"[file_logger] Failed to log new agent: {e}")

    def log_event(self, *args, **kwargs) -> None:
        if not self.logger:
            return
        return super().log_event(*args, **kwargs)

    def log_new_wrapper(self, *args, **kwargs) -> None:
        if not self.logger:
            return
        return super().log_new_wrapper(*args, **kwargs)

    def log_new_client(self, *args, **kwargs) -> None:
        if not self.logger:
            return
        return super().log_new_client(*args, **kwargs)

    def __repr__(self) -> str:
        return self.session.__repr__()
