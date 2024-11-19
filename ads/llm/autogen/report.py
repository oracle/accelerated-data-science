# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import logging
import os
from typing import Optional

import autogen
import autogen.runtime_logging
from ads.llm.autogen.oci_logger import OCIFileLogger


logger = logging.getLogger(__name__)


class AutoGenLoggingException(Exception):
    pass


def start_logging(
    log_dir: str, session_id: Optional[str] = None, auth: Optional[dict] = None
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
        If session_id is None, a new UUID4 will be generated.
        The session ID will be used as the log filename.
    auth: dict, optional
        Dictionary containing the OCI authentication config and signer.
        This is only used if log_dir is on object storage.
        If auth is None, `ads.common.auth.default_signer()` will be used.

    Returns
    -------
    str
        Session ID
    """
    autogen_logger = autogen.runtime_logging.autogen_logger
    if autogen_logger is None:
        autogen_logger = OCIFileLogger(
            log_dir=log_dir, session_id=session_id, auth=auth
        )
    elif isinstance(autogen_logger, OCIFileLogger):
        autogen_logger.new_session(log_dir=log_dir, session_id=session_id, auth=auth)
    elif autogen.runtime_logging.is_logging:
        raise AutoGenLoggingException(
            "AutoGen is currently logging with a different logger. "
            "Only one logger can be active at a time. "
            "Please call `autogen.runtime_logging.stop()` to stop logging "
            "before starting a new session."
        )
    else:
        logger.warning("Replacing AutoGen logger with OCIFileLogger...")
        autogen_logger = OCIFileLogger(log_dir=log_dir, session_id=session_id)
    return autogen.runtime_logging.start(logger=autogen_logger)


def stop_logging(
    report_dir: str = None, return_par_uri: bool = False, **kwargs
) -> Optional[str]:
    """Stops the logging session.

    Parameters
    ----------
    report_dir : str, optional
        Directory for saving the session report, by default None.
        If `report_dir` is None, no report will be created.
    return_par_uri: bool, optional
        For report_dir on OCI object storage only,
        whether to create and return a pre-authenticated uri for the report.
        Defaults to False.

    Returns
    -------
    str
        The full filename of the report, if `report_dir` is provided.
        Otherwise, None.

    """
    autogen.runtime_logging.stop()
    logger = autogen.runtime_logging.autogen_logger

    if not report_dir:
        if isinstance(logger, OCIFileLogger):
            return logger.session
        else:
            return None

    if not isinstance(logger, OCIFileLogger):
        raise NotImplementedError("The logger does not support report generation.")
    report_file = os.path.join(report_dir, f"{logger.session_id}.html")
    logger.session.create_report(
        report_file=report_file, return_par_uri=return_par_uri, **kwargs
    )
    return logger.session
