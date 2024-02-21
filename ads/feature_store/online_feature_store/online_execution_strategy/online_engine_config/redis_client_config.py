#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


try:
    import redis
except ImportError:
    pass


class RedisClientConfig:
    def __init__(self, host="localhost", port=6379, password=None, db=0):
        """
        Initialize the Redis client configuration.

        Parameters:
            - host (str): The Redis server address.
            - port (int): The Redis server port.
            - password (str): The Redis password for authentication.
            - db (int): The Redis database index.

        Example:
            config = RedisClientConfig(host='localhost', port=6379, password='your_password')
            redis_client = config.get_client()
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db

    def get_client(self):
        """
        Get an instance of the Redis client configured based on the provided parameters.

        Returns:
            Redis: An instance of the Redis client.
        """
        return redis.StrictRedis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            encoding="utf-8",
            ssl=True,
            ssl_cert_reqs=None,
            decode_responses=True,  # Decodes responses from bytes to strings
        )
