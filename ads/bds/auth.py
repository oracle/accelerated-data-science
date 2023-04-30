#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
import subprocess
from contextlib import contextmanager


DEFAULT_KRB5_CONFIG_PATH = "~/.bds_config/krb5.conf"
KRB5_CONFIG = "KRB5_CONFIG"


class KRB5KinitError(Exception):   # pragma: no cover
    """KRB5KinitError class when kinit -kt command failed to generate cached ticket with the keytab file and the krb5 config file."""

    pass


def has_kerberos_ticket():
    """Whether kerberos cache ticket exists."""
    return True if subprocess.call(["klist", "-s"]) == 0 else False


def init_ccache_with_keytab(principal: str, keytab_file: str) -> None:
    """Initialize credential cache using keytab file.

    Parameters
    ----------
    principal: str
        The unique identity to which Kerberos can assign tickets.
    keytab_path: str
        Path to your keytab file.

    Returns
    -------
    None
        Nothing.
    """
    cmd = "kinit -kt %(keytab_file)s  %(principal)s"
    args = {}

    args["principal"] = principal
    args["keytab_file"] = keytab_file

    kinit_proc = subprocess.Popen((cmd % args).split(), stderr=subprocess.PIPE)
    stdout_data, stderr_data = kinit_proc.communicate()

    if kinit_proc.returncode > 0:
        raise KRB5KinitError(stderr_data)


@contextmanager
def krbcontext(
    principal: str, keytab_path: str, kerb5_path: str = DEFAULT_KRB5_CONFIG_PATH
) -> None:
    """A context manager for Kerberos-related actions.
    It provides a Kerberos context that you can put code inside.
    It will initialize credential cache automatically with keytab if no cached ticket exists.
    Otherwise, does nothing.

    Parameters
    ----------
    principal: str
        The unique identity to which Kerberos can assign tickets.
    keytab_path: str
        Path to your keytab file.
    kerb5_path: (str, optional).
        Path to your krb5 config file.

    Returns
    -------
    None
        Nothing.

    Examples
    --------
    >>> from ads.bds.auth import krbcontext
    >>> from pyhive import hive
    >>> with krbcontext(principal = "your_principal", keytab_path = "your_keytab_path"):
    >>>    hive_cursor = hive.connect(host="your_hive_host",
    ...                    port="your_hive_port",
    ...                    auth='KERBEROS',
    ...                    kerberos_service_name="hive").cursor()
    """
    refresh_ticket(principal=principal, keytab_path=keytab_path, kerb5_path=kerb5_path)
    yield


def refresh_ticket(
    principal: str, keytab_path: str, kerb5_path: str = DEFAULT_KRB5_CONFIG_PATH
) -> None:
    """generate new cached ticket based on the principal and keytab file path.

    Parameters
    ----------
    principal: str
        The unique identity to which Kerberos can assign tickets.
    keytab_path: str
        Path to your keytab file.
    kerb5_path: (str, optional).
        Path to your krb5 config file.

    Returns
    -------
    None
        Nothing.

    Examples
    --------
    >>> from ads.bds.auth import refresh_ticket
    >>> from pyhive import hive
    >>> refresh_ticket(principal = "your_principal", keytab_path = "your_keytab_path")
    >>> hive_cursor = hive.connect(host="your_hive_host",
    ...                    port="your_hive_port",
    ...                    auth='KERBEROS',
    ...                    kerberos_service_name="hive").cursor()
    """
    keytab_path = os.path.abspath(os.path.expanduser(keytab_path))
    os.environ[KRB5_CONFIG] = os.path.abspath(os.path.expanduser(kerb5_path))
    if not os.path.exists(os.environ[KRB5_CONFIG]):
        raise FileNotFoundError(f"krb5 config file not found in {kerb5_path}.")
    if not has_kerberos_ticket():
        init_ccache_with_keytab(principal, keytab_path)
