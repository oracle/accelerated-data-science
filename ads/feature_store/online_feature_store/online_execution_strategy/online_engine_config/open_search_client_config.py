#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from elasticsearch import Elasticsearch


class OpenSearchClientConfig:
    def __init__(self, host, scheme="https", verify_certs=True):
        """
        Initialize the Elasticsearch client configuration.

        Parameters:
            - hosts (str or list): The Elasticsearch server address. Can be a single string or a list of strings.
            - username (str): The Elasticsearch username for authentication.
            - password (str): The Elasticsearch password for authentication.
            - scheme (str): The scheme used for connecting to Elasticsearch ('http' or 'https').
            - verify_certs (bool): Whether to verify SSL certificates.

        Example:
            config = OpenSearchClientConfig(hosts='localhost', username='elastic', password='<>')
            es = config.get_client()
        """
        self.host = host
        self.scheme = scheme
        self.verify_certs = verify_certs

    def get_client(self, http_auth: tuple[str, str]):
        """
        Get an instance of the Elasticsearch client configured based on the provided parameters.

        Returns:
            Elasticsearch: An instance of the Elasticsearch client.
        """
        return Elasticsearch(
            hosts=f"{self.scheme}://{self.host}:9200",
            basic_auth=http_auth,
            verify_certs=self.verify_certs,
            ssl_show_warn=False,
        )
