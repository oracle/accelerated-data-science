# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import logging
import os
from typing import Optional

from ads.llm.autogen.v02 import runtime_logging
from ads.llm.autogen.v02.session_logger import SessionLogger


logger = logging.getLogger(__name__)


class AutoGenLoggingException(Exception):
    pass


def start_logging(
    log_dir: str,
    report_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    auth: Optional[dict] = None,
    report_par_uri=False,
    **kwargs,
) -> str:
    """Starts a new logging session.
    Each thread can only have one logging session.

    AutoGen saves the logger as global variable. Only one logger can be active at a time.
    If you are using other loggers like AgentOps, an exception will be raised.

    Parameters
    ----------
    log_dir : str
        The location to store the logs.
    session_id : str, optional
        Session ID for identifying the session, by default None.
        The session ID will be used as the log filename.
        If session_id is None, a new UUID4 will be generated.
        To resume a session, use a previously generated session_id.

    auth: dict, optional
        Dictionary containing the OCI authentication config and signer.
        This is only used if log_dir is on object storage.
        If auth is None, `ads.common.auth.default_signer()` will be used.

    Returns
    -------
    str
        Session ID
    """
    autogen_logger = SessionLogger(
        log_dir=log_dir,
        report_dir=report_dir,
        session_id=session_id,
        auth=auth,
        report_par_uri=report_par_uri,
        par_kwargs=kwargs,
    )
    return runtime_logging.start(logger=autogen_logger)


def stop_logging():
    """Stops the logging session."""
    return runtime_logging.stop()
