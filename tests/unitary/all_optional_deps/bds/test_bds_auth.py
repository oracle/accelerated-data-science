#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import subprocess
from unittest import mock

from ads.bds.auth import (
    has_kerberos_ticket,
    init_ccache_with_keytab,
    refresh_ticket,
    krbcontext,
)


def test_has_kerberos_ticket():
    with mock.patch("subprocess.call", return_value=0):
        assert has_kerberos_ticket()


def test_krbcontext():
    keytab_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "bds", "training.keytab"
    )

    def mock_refresh_ticket(principal, keytab_path, kerb5_path):
        os.environ["KRB5_CONFIG"] = "fake_value"

    with mock.patch("ads.bds.auth.refresh_ticket", side_effect=mock_refresh_ticket):
        with krbcontext("fake_principal", keytab_path=keytab_file):
            assert os.environ["KRB5_CONFIG"] == "fake_value"


def test_init_ccache_with_keytab():
    class KinitProc:
        def communicate(self):
            return 0, 0

        @property
        def returncode(self):
            return 0

    def mock_subprocess_popen():
        return KinitProc()

    with mock.patch("subprocess.Popen", return_value=mock_subprocess_popen()):
        keytab_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "bds", "training.keytab"
        )
        init_ccache_with_keytab("fake_principal", keytab_file)


def test_refresh_ticket(tmpdir):
    cache_location = os.path.join(tmpdir, "cache_location", "cache_ticket")
    krb5_location = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "bds", "krb5.conf"
    )
    keytab_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "bds", "training.keytab"
    )
    if "KRB5_CONFIG" in os.environ:
        del os.environ["KRB5_CONFIG"]

    def mock_init_ccache_with_keytab():
        with open(cache_location, "wb") as fw:
            fw.write(f"cached ticket.")

    with mock.patch("ads.bds.auth.has_kerberos_ticket", return_value=True):
        with mock.patch(
            "ads.bds.auth.init_ccache_with_keytab",
            side_effect=mock_init_ccache_with_keytab,
        ):
            refresh_ticket(
                "fake_principal",
                keytab_path=keytab_file,
                kerb5_path=krb5_location,
            )
            assert os.environ["KRB5_CONFIG"] == krb5_location
