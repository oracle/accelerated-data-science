# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.
import logging
from sqlite3 import Connection
from typing import Any, Dict, List, Optional

import autogen.runtime_logging
from autogen.logger.base_logger import BaseLogger
from autogen.logger.logger_factory import LoggerFactory

module_logger = logging.getLogger(__name__)


class LoggerManager(BaseLogger):

    def __init__(self) -> None:
        self.loggers: List[BaseLogger] = []
        super().__init__()

    def add_logger(self, logger: BaseLogger) -> None:
        """Adds a new AutoGen logger."""
        self.loggers.append(logger)

    def call_loggers(self, method, *args, **kwargs) -> None:
        for autogen_logger in self.loggers:
            getattr(autogen_logger, method)(*args, **kwargs)

    def start(self) -> str:
        return self.call_loggers("start")

    def stop(self) -> None:
        return self.call_loggers("stop")

    def get_connection(self) -> None | Connection:
        return self.call_loggers("get_connection")

    def log_chat_completion(self, *args, **kwargs) -> None:
        return self.call_loggers("log_chat_completion", *args, **kwargs)

    def log_new_agent(self, *args, **kwargs) -> None:
        return self.call_loggers("log_new_agent", *args, **kwargs)

    def log_event(self, *args, **kwargs) -> None:
        return self.call_loggers("log_event", *args, **kwargs)

    def log_new_wrapper(self, *args, **kwargs) -> None:
        return self.call_loggers("log_new_wrapper", *args, **kwargs)

    def log_new_client(self, *args, **kwargs) -> None:
        return self.call_loggers("log_new_client", *args, **kwargs)

    def log_function_use(self, *args, **kwargs) -> None:
        return self.call_loggers("log_function_use", *args, **kwargs)

    def __repr__(self) -> str:
        return "\n\n".join(
            [
                f"{str(logger.__class__)}:\n{logger.__repr__()}"
                for logger in self.loggers
            ]
        )


def start(
    logger: Optional[BaseLogger] = None,
    logger_type: str = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    if logger and logger_type:
        raise ValueError(
            "Please specify only logger(%s) or logger_type(%s).", logger, logger_type
        )

    # Check if a logger is already configured
    autogen_logger = autogen.runtime_logging.autogen_logger
    if not autogen_logger:
        # No logger is configured
        logger_manager = LoggerManager()
    elif not isinstance(autogen_logger, LoggerManager):
        # Logger is configured but it is not via ADS
        module_logger.warning(
            "AutoGen is already configured with %s", str(autogen_logger)
        )
        logger_manager = LoggerManager()
        logger_manager.add_logger(autogen_logger)
    else:
        # Logger is already configured with ADS
        logger_manager = autogen_logger

    # Add AutoGen logger
    if logger:
        autogen_logger = logger
    else:
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
    finally:
        return session_id


def stop():
    autogen.runtime_logging.stop()
    return autogen.runtime_logging.autogen_logger
