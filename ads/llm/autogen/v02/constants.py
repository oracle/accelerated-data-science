# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


class Events:
    KEY = "event_name"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    NEW_AGENT = "new_agent"
    NEW_CLIENT = "new_client"
    SESSION_START = "logging_session_start"
    SESSION_STOP = "logging_session_stop"
