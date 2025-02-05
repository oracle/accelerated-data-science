#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test OCI Data Science Model Deployment Endpoint."""

import pytest
import sys
from unittest.mock import MagicMock, patch

if sys.version_info < (3, 9):
    pytest.skip(allow_module_level=True)

from ads.llm import OCIDataScienceEmbedding


@patch("ads.llm.OCIDataScienceEmbedding._embed_with_retry")
def test_embed_documents(mock_embed_with_retry) -> None:
    """Test valid call to oci model deployment endpoint."""
    expected_output = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    result = MagicMock()
    result.json = MagicMock(
        return_value={
            "data": [{"embedding": expected_output}],
        }
    )
    mock_embed_with_retry.return_value = result
    endpoint = "https://MD_OCID/predict"
    documents = ["Hello", "World"]

    embeddings = OCIDataScienceEmbedding(
        endpoint=endpoint,
    )

    output = embeddings.embed_documents(documents)
    assert output == expected_output


@patch("ads.llm.OCIDataScienceEmbedding._embed_with_retry")
def test_embed_query(mock_embed_with_retry) -> None:
    """Test valid call to oci model deployment endpoint."""
    expected_output = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    result = MagicMock()
    result.json = MagicMock(
        return_value={
            "data": [{"embedding": expected_output}],
        }
    )
    mock_embed_with_retry.return_value = result
    endpoint = "https://MD_OCID/predict"
    query = "Hello world"

    embeddings = OCIDataScienceEmbedding(
        endpoint=endpoint,
    )

    output = embeddings.embed_query(query)
    assert output == expected_output[0]
