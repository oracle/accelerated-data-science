# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class Events:
    KEY = "event_name"

    EXCEPTION = "exception"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    NEW_AGENT = "new_agent"
    NEW_CLIENT = "new_client"
    RECEIVED_MESSAGE = "received_message"
    SESSION_START = "logging_session_start"
    SESSION_STOP = "logging_session_stop"
