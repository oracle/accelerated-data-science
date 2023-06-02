#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64


class Base64EncoderDecoder:
    """
    A utility class for encoding and decoding data in Base64 format.

    Usage:
    To encode a string:
    ```
    encoded_data = Base64EncoderDecoder.encode("Hello, World!")
    ```

    To decode an encoded string:
    ```
    decoded_data = Base64EncoderDecoder.decode(encoded_data)
    ```

    Attributes:
    -----------
    None

    Methods:
    --------
    encode(raw_data: str) -> str:
        Encodes the provided raw data into Base64 format and returns the encoded data as a string.

    decode(encoded_data: str) -> str:
        Decodes the provided Base64 data into its original format and returns the decoded data as a string.
    """

    @classmethod
    def encode(cls, raw_data: str) -> str:
        """Encodes raw data into Base64 format.

        Parameters:
        ----------
        raw_data: str
            The raw data that needs to be encoded.

        Returns:
        -------
        str:
            Base64-encoded data as a string.
        """
        byte_code = raw_data.encode("ascii")
        base64_bytes = base64.b64encode(byte_code)
        return base64_bytes.decode("ascii")

    @classmethod
    def decode(cls, encoded_data: str) -> str:
        """Decodes Base64 data into its original format.

        Parameters:
        ----------
        encoded_data: str
            The Base64-encoded data that needs to be decoded.

        Returns:
        -------
        str:
            Decoded data as a string.
        """
        base64_bytes = encoded_data.encode("ascii")
        message_bytes = base64.b64decode(base64_bytes)
        return message_bytes.decode("ascii")
