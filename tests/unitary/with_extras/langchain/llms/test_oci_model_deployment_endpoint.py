#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test OCI Data Science Model Deployment Endpoint."""

import sys
from unittest import mock
import pytest
from requests.exceptions import HTTPError
from ads.llm import OCIModelDeploymentTGI, OCIModelDeploymentVLLM

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires Python 3.9 or higher"
)


CONST_MODEL_NAME = "odsc-vllm"
CONST_ENDPOINT = "https://oci.endpoint/ocid/predict"
CONST_PROMPT = "This is a prompt."
CONST_COMPLETION = "This is a completion."
CONST_COMPLETION_RESPONSE = {
    "choices": [
        {
            "index": 0,
            "text": CONST_COMPLETION,
            "logprobs": 0.1,
            "finish_reason": "length",
        }
    ],
}
CONST_COMPLETION_RESPONSE_TGI = {"generated_text": CONST_COMPLETION}
CONST_STREAM_TEMPLATE = (
    'data: {"id":"","object":"text_completion","created":123456,'
    + '"choices":[{"index":0,"text":"<TOKEN>","finish_reason":""}]}'
)
CONST_STREAM_RESPONSE = (
    CONST_STREAM_TEMPLATE.replace("<TOKEN>", " " + word).encode()
    for word in CONST_COMPLETION.split(" ")
)

CONST_ASYNC_STREAM_TEMPLATE = (
    '{"id":"","object":"text_completion","created":123456,'
    + '"choices":[{"index":0,"text":"<TOKEN>","finish_reason":""}]}'
)
CONST_ASYNC_STREAM_RESPONSE = (
    CONST_ASYNC_STREAM_TEMPLATE.replace("<TOKEN>", " " + word).encode()
    for word in CONST_COMPLETION.split(" ")
)


def mocked_requests_post(self, **kwargs):
    """Method to mock post requests"""

    class MockResponse:
        """Represents a mocked response."""

        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code

        def raise_for_status(self):
            """Mocked raise for status."""
            if 400 <= self.status_code < 600:
                raise HTTPError("", response=self)

        def json(self):
            """Returns mocked json data."""
            return self.json_data

        def iter_lines(self, chunk_size=4096):
            """Returns a generator of mocked streaming response."""
            return CONST_STREAM_RESPONSE

        @property
        def text(self):
            return ""

    payload = kwargs.get("json")
    if "inputs" in payload:
        prompt = payload.get("inputs")
        is_tgi = True
    else:
        prompt = payload.get("prompt")
        is_tgi = False

    if prompt == CONST_PROMPT:
        if is_tgi:
            return MockResponse(json_data=CONST_COMPLETION_RESPONSE_TGI)
        return MockResponse(json_data=CONST_COMPLETION_RESPONSE)

    return MockResponse(
        json_data={},
        status_code=404,
    )


async def mocked_async_streaming_response(*args, **kwargs):
    """Returns mocked response for async streaming."""
    for item in CONST_ASYNC_STREAM_RESPONSE:
        yield item


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_invoke_vllm(mock_post, mock_auth) -> None:
    """Tests invoking vLLM endpoint."""
    llm = OCIModelDeploymentVLLM(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    output = llm.invoke(CONST_PROMPT)
    assert output == CONST_COMPLETION


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_stream_tgi(mock_post, mock_auth) -> None:
    """Tests streaming with TGI endpoint using OpenAI spec."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    output = ""
    count = 0
    for chunk in llm.stream(CONST_PROMPT):
        output += chunk
        count += 1
    assert count == 4
    assert output.strip() == CONST_COMPLETION


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_generate_tgi(mock_post, mock_auth) -> None:
    """Tests invoking TGI endpoint using TGI generate spec."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, api="/generate", model=CONST_MODEL_NAME
    )
    output = llm.invoke(CONST_PROMPT)
    assert output == CONST_COMPLETION


@pytest.mark.asyncio
@pytest.mark.requires("ads")
@mock.patch(
    "ads.common.auth.default_signer", return_value=dict(signer=mock.MagicMock())
)
@mock.patch(
    "langchain_community.utilities.requests.Requests.apost",
    mock.MagicMock(),
)
async def test_stream_async(mock_auth):
    """Tests async streaming."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    with mock.patch.object(
        llm,
        "_aiter_sse",
        mock.MagicMock(return_value=mocked_async_streaming_response()),
    ):

        chunks = [chunk async for chunk in llm.astream(CONST_PROMPT)]
    assert "".join(chunks).strip() == CONST_COMPLETION
