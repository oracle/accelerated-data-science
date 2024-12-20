# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
import traceback
from sqlite3 import Connection
from typing import Any, Dict, List, Optional

import autogen.runtime_logging
from autogen.logger.base_logger import BaseLogger
from autogen.logger.logger_factory import LoggerFactory

logger = logging.getLogger(__name__)


class LoggerManager(BaseLogger):
    """Manages multiple AutoGen loggers."""

    def __init__(self) -> None:
        self.loggers: List[BaseLogger] = []
        super().__init__()

    def add_logger(self, logger: BaseLogger) -> None:
        """Adds a new AutoGen logger."""
        self.loggers.append(logger)

    def _call_loggers(self, method: str, *args, **kwargs) -> None:
        """Calls the specific method on each AutoGen logger in self.loggers."""
        for autogen_logger in self.loggers:
            try:
                getattr(autogen_logger, method)(*args, **kwargs)
            except Exception as e:
                # Catch the logging exception so that the program will not be interrupted.
                logger.error(
                    "Failed to %s with %s: %s",
                    method,
                    autogen_logger.__class__.__name__,
                    str(e),
                )
                logger.debug(traceback.format_exc())

    def start(self) -> str:
        """Starts all loggers."""
        return self._call_loggers("start")

    def stop(self) -> None:
        self._call_loggers("stop")
        # Remove the loggers once they are stopped.
        self.loggers = []

    def get_connection(self) -> None | Connection:
        return self._call_loggers("get_connection")

    def log_chat_completion(self, *args, **kwargs) -> None:
        return self._call_loggers("log_chat_completion", *args, **kwargs)

    def log_new_agent(self, *args, **kwargs) -> None:
        return self._call_loggers("log_new_agent", *args, **kwargs)

    def log_event(self, *args, **kwargs) -> None:
        return self._call_loggers("log_event", *args, **kwargs)

    def log_new_wrapper(self, *args, **kwargs) -> None:
        return self._call_loggers("log_new_wrapper", *args, **kwargs)

    def log_new_client(self, *args, **kwargs) -> None:
        return self._call_loggers("log_new_client", *args, **kwargs)

    def log_function_use(self, *args, **kwargs) -> None:
        return self._call_loggers("log_function_use", *args, **kwargs)

    def __repr__(self) -> str:
        return "\n\n".join(
            [
                f"{str(logger.__class__)}:\n{logger.__repr__()}"
                for logger in self.loggers
            ]
        )


def start(
    autogen_logger: Optional[BaseLogger] = None,
    logger_type: str = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Starts logging with AutoGen logger.
    Specify your custom autogen_logger, or the logger_type and config to use a built-in logger.

    Parameters
    ----------
    autogen_logger : BaseLogger, optional
        An AutoGen logger, which should be a subclass of autogen.logger.base_logger.BaseLogger.
    logger_type : str, optional
        Logger type, which can be a built-in AutoGen logger type ("file", or "sqlite"), by default None.
    config : dict, optional
        Configurations for the built-in AutoGen logger, by default None

    Returns
    -------
    str
        A unique session ID returned from starting the logger.

    """
    if autogen_logger and logger_type:
        raise ValueError(
            "Please specify only autogen_logger(%s) or logger_type(%s).",
            autogen_logger,
            logger_type,
        )

    # Check if a logger is already configured
    existing_logger = autogen.runtime_logging.autogen_logger
    if not existing_logger:
        # No logger is configured
        logger_manager = LoggerManager()
    elif isinstance(existing_logger, LoggerManager):
        # Logger is already configured with ADS
        logger_manager = existing_logger
    else:
        # Logger is configured but it is not via ADS
        logger.warning("AutoGen is already configured with %s", str(existing_logger))
        logger_manager = LoggerManager()
        logger_manager.add_logger(existing_logger)

    # Add AutoGen logger
    if not autogen_logger:
        autogen_logger = LoggerFactory.get_logger(
            logger_type=logger_type, config=config
        )
    logger_manager.add_logger(autogen_logger)

    try:
        session_id = autogen_logger.start()
        autogen.runtime_logging.is_logging = True
        autogen.runtime_logging.autogen_logger = logger_manager
    except Exception as e:
        logger.error(f"Failed to start logging: {e}")
    return session_id


def stop(*loggers) -> BaseLogger:
    """Stops AutoGen logger.
    If loggers are managed by LoggerManager,
    you may specify one or more loggers to be stopped.
    If no logger is specified, all loggers will be stopped.
    Stopped loggers will be removed from the LoggerManager.
    """
    autogen_logger = autogen.runtime_logging.autogen_logger
    if isinstance(autogen_logger, LoggerManager) and loggers:
        for logger in loggers:
            logger.stop()
            if logger in autogen_logger.loggers:
                autogen_logger.loggers.remove(logger)
    else:
        autogen.runtime_logging.stop()
    return autogen_logger


def get_loggers() -> List[BaseLogger]:
    """Gets a list of existing AutoGen loggers."""
    autogen_logger = autogen.runtime_logging.autogen_logger
    if isinstance(autogen_logger, LoggerManager):
        return autogen_logger.loggers
    return [autogen_logger]
