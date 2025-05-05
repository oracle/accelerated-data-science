#!/usr/bin/env python

# Copyright (c) 2021, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
This module forwards authentication utilities to the new 'oracle-ads-auth' package
to preserve backward compatibility.

All functionality is now implemented in the `ads.auth` namespace from the
`oracle-ads-auth` package.
"""

import inspect
import warnings

# Inspect the stack to check if the import comes from outside `ads`
_stack = inspect.stack()
if all("ads/" not in frame.filename.replace("\\", "/") for frame in _stack[1:]):
    warnings.warn(
        "'ads.common.auth' is deprecated and will be removed in a future release.\n"
        "Please update your imports to use 'ads.auth' from the new 'oracle-ads-auth' package.",
        DeprecationWarning,
        stacklevel=2,
    )

from ads.auth import (
    AuthContext,
    AuthFactory,
    AuthState,
    AuthType,
    SecurityToken,
    SecurityTokenError,
    api_keys,
    create_signer,
    default_signer,
    register_user_agent_hook,
    resource_principal,
    security_token,
    set_auth,
)

__all__ = [
    "set_auth",
    "default_signer",
    "create_signer",
    "api_keys",
    "resource_principal",
    "security_token",
    "AuthType",
    "AuthContext",
    "register_user_agent_hook",
    "SecurityToken",
    "SecurityTokenError",
    "AuthState",
    "AuthFactory",
]
