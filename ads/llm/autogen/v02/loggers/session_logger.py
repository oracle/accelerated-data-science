# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import importlib
import logging
import os
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import autogen
import fsspec
import oci
from autogen import Agent, ConversableAgent, GroupChatManager, OpenAIWrapper
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
from ads.llm.autogen.constants import Events
from ads.llm.autogen.reports.data import (
    AgentData,
    LLMCompletionData,
    LogRecord,
    ToolCallData,
)
from ads.llm.autogen.reports.session import SessionReport
from ads.llm.autogen.v02 import runtime_logging
from ads.llm.autogen.v02.log_handlers.oci_file_handler import OCIFileHandler
from ads.llm.autogen.v02.loggers.utils import (
    serialize,
    serialize_response,
)

logger = logging.getLogger(__name__)


CONST_REPLY_FUNC_NAME = "reply_func_name"


@dataclass
class LoggingSession:
    """Represents a logging session for a specific thread."""

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
        """HTML report path of the logging session.
        If the a pre-authenticated link is generated for the report,
        the pre-authenticated link will be returned.

        If the report is saved to OCI object storage, the URI will be return.
        If the report is saved locally, the local path will be return.
        If there is no report generated, `None` will be returned.
        """
        if self.par_uri:
            return self.par_uri
        elif self.report_file:
            return self.report_file
        return None

    def __repr__(self) -> str:
        """Shows the link to report if it is available, otherwise shows the log file link."""
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
        # Log only if started is True
        self.started = False

        # Keep track of last check_termination_and_human_reply for calculating tool call duration
        # This will be a dictionary mapping the IDs of the agents to their last timestamp
        # of check_termination_and_human_reply
        self.last_agent_checks = {}

    @property
    def logger(self) -> Optional[logging.Logger]:
        """Logger for the thread.

        This property is used to determine whether the log should be saved.
        No log will be saved if the logger is None.
        """
        if not self.started:
            return None
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
    def report(self) -> Optional[str]:
        """Report path/link for the  session."""
        return self.session.report

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
        report_link = self.session.create_report(
            report_file=report_file, return_par_uri=self.report_par_uri, **kwargs
        )
        print(f"ADS AutoGen Session Report: {report_link}")
        return report_link

    def new_record(self, event_name: str, source: Any = None) -> LogRecord:
        """Initialize a new log record.

        The record is not logged until `self.log()` is called.
        """
        record = LogRecord(
            session_id=self.session_id,
            thread_id=threading.get_ident(),
            timestamp=get_current_ts(),
            event_name=event_name,
        )
        if source:
            record.source_id = id(source)
            record.source_name = str(source.name) if hasattr(source, "name") else source
        return record

    def log(self, record: LogRecord) -> None:
        """Logs a record.

        Parameters
        ----------
        data : dict
            Data to be logged.
        """
        # Do nothing if there is no logger for the thread.
        if not self.logger:
            return

        try:
            self.logger.info(record.to_string())
        except Exception:
            self.logger.info("Failed to log %s", record.event_name)

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
        self.started = True
        self.log_event(source=self, name=Events.SESSION_START, environment=envs)
        return self.session_id

    def stop(self) -> None:
        """Stops the logging session."""
        self.log_event(source=self, name=Events.SESSION_STOP)
        super().stop()
        self.started = False
        if self.report_dir:
            try:
                self.generate_report()
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
        Logs a chat completion.
        """
        if not self.logger:
            return

        record = self.new_record(event_name=Events.LLM_CALL, source=source)
        record.data = LLMCompletionData(
            invocation_id=str(invocation_id),
            request=serialize(request),
            response=serialize_response(response),
            start_time=start_time,
            end_time=get_current_ts(),
            cost=cost,
            is_cached=is_cached,
        )
        record.kwargs = {
            "client_id": client_id,
            "wrapper_id": wrapper_id,
        }

        self.log(record)

    def log_function_use(
        self, source: Union[str, Agent], function: F, args: Dict[str, Any], returns: Any
    ) -> None:
        """
        Logs a registered function(can be a tool) use from an agent or a string source.
        """
        if not self.logger:
            return

        source_id = id(source)
        if source_id in self.last_agent_checks:
            start_time = self.last_agent_checks[source_id]
        else:
            start_time = get_current_ts()

        record = self.new_record(Events.TOOL_CALL, source=source)
        record.data = ToolCallData(
            tool_name=function.__name__,
            start_time=start_time,
            end_time=record.timestamp,
            agent_name=str(source.name) if hasattr(source, "name") else source,
            agent_module=source.__module__,
            agent_class=source.__class__.__name__,
            input_args=safe_serialize(args),
            returns=safe_serialize(returns),
        )

        self.log(record)

    def log_new_agent(
        self, agent: ConversableAgent, init_args: Dict[str, Any] = {}
    ) -> None:
        """
        Logs a new agent instance.
        """
        if not self.logger:
            return

        record = self.new_record(event_name=Events.NEW_AGENT, source=agent)
        record.data = AgentData(
            agent_name=(
                agent.name
                if hasattr(agent, "name") and agent.name is not None
                else str(agent)
            ),
            agent_module=agent.__module__,
            agent_class=agent.__class__.__name__,
            is_manager=isinstance(agent, GroupChatManager),
        )
        record.kwargs = {
            "wrapper_id": serialize(
                agent.client.wrapper_id
                if hasattr(agent, "client") and agent.client is not None
                else ""
            ),
            "args": serialize(init_args),
        }
        self.log(record)

    def log_event(
        self, source: Union[str, Agent], name: str, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Logs an event.
        """
        record = self.new_record(event_name=name)
        record.source_id = id(source)
        record.source_name = str(source.name) if hasattr(source, "name") else source
        record.kwargs = kwargs
        if isinstance(source, Agent):
            if (
                CONST_REPLY_FUNC_NAME in kwargs
                and kwargs[CONST_REPLY_FUNC_NAME] == "check_termination_and_human_reply"
            ):
                self.last_agent_checks[record.source_id] = record.timestamp
            record.data = AgentData(
                agent_name=record.source_name,
                agent_module=source.__module__,
                agent_class=source.__class__.__name__,
                is_manager=isinstance(source, GroupChatManager),
            )
        self.log(record)

    def log_new_wrapper(self, *args, **kwargs) -> None:
        # Do not log new wrapper.
        # This is not used at the moment.
        return

    def log_new_client(
        self,
        client,
        wrapper: OpenAIWrapper,
        init_args: Dict[str, Any],
    ) -> None:
        if not self.logger:
            return

        record = self.new_record(event_name=Events.NEW_CLIENT)
        # init_args may contain credentials like api_key
        record.kwargs = {
            "client_id": id(client),
            "wrapper_id": id(wrapper),
            "class": client.__class__.__name__,
            "args": serialize(init_args),
        }

        self.log(record)

    def __repr__(self) -> str:
        return self.session.__repr__()

    def __enter__(self) -> "SessionLogger":
        """Starts the session logger

        Returns
        -------
        SessionLogger
            The session logger
        """
        runtime_logging.start(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """Stops the session logger."""
        if exc_type:
            record = self.new_record(event_name=Events.EXCEPTION)
            record.kwargs = {
                "exc_type": exc_type.__name__,
                "exc_value": str(exc_value),
                "traceback": "".join(traceback.format_tb(tb)),
                "locals": serialize(tb.tb_frame.f_locals),
            }
            self.log(record)
        runtime_logging.stop(self)
