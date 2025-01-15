#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test OCI Data Science Model Deployment Endpoint."""

from unittest.mock import MagicMock, patch
from ads.llm.langchain.plugins.embeddings.oci_data_science_model_deployment_endpoint import (
    OCIModelDeploymentEndpointEmbeddings,
)


@patch("ads.llm.OCIModelDeploymentEndpointEmbeddings._embed_with_retry")
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

    embeddings = OCIModelDeploymentEndpointEmbeddings(
        endpoint=endpoint,
    )

    output = embeddings.embed_documents(documents)
    assert output == expected_output


@patch("ads.llm.OCIModelDeploymentEndpointEmbeddings._embed_with_retry")
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

    embeddings = OCIModelDeploymentEndpointEmbeddings(
        endpoint=endpoint,
    )

    output = embeddings.embed_query(query)
    assert output == expected_output[0]
