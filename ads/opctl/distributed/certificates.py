#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import oci
import os


def download_certificates(auth, ca_file_name, cert_file_name, key_file_name):
    """ """
    client = oci.certificates.CertificatesClient(**auth)
    # Download Cert and Private Key
    print("Fetching certificate details")
    response = client.get_certificate_bundle(
        certificate_id=os.environ["OCI__CERTIFICATE_OCID"],
        certificate_bundle_type="CERTIFICATE_CONTENT_WITH_PRIVATE_KEY",
    )
    with open(cert_file_name, "w") as f:
        print(f"writing {cert_file_name}")
        f.write(response.data.certificate_pem)
    with open(key_file_name, "w") as f:
        print(f"writing {key_file_name}")
        f.write(response.data.private_key_pem)
    # Download CA Cert
    response = client.get_certificate_authority_bundle(
        certificate_authority_id=os.environ["OCI__CERTIFICATE_AUTHORITY_OCID"]
    )
    with open(ca_file_name, "w") as f:
        print(f"writing {ca_file_name}")
        f.write(response.data.cert_chain_pem)
