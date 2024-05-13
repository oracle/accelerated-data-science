#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ads.aqua.playground.const import MessageRate, MessageRole, Status
from ads.aqua.playground.entities import (
    Message,
    ModelInfo,
    SearchId,
    Session,
    Thread,
    VLLModelParams,
)


class TestSearchId:
    def test_parse_valid_model_id(self):
        # Test parsing a string that contains 'ocid' as a model ID
        model_id = "test_id"
        result = SearchId.parse(model_id)
        assert result.model_id == model_id
        assert result.record_id is None

    def test_parse_valid_record_id(self):
        # Test parsing a string that is a numeric value as a record ID
        record_id = "12345"
        result = SearchId.parse(record_id)
        assert result.record_id == int(record_id)
        assert result.model_id is None

    def test_parse_invalid_id(self):
        # Test parsing an invalid string which should raise ValueError
        invalid_id = "invalid"
        with pytest.raises(ValueError):
            SearchId.parse(invalid_id)


class TestVLLModelParams:
    def test_default_params(self):
        # Test the default initialization of VLLModelParams
        params = VLLModelParams()
        assert params.model == "/opt/ds/model/deployed_model"
        assert params.max_tokens == 2048
        assert params.temperature == 0.7
        assert params.top_p == 1.0
        assert params.frequency_penalty == 0.0
        assert params.presence_penalty == 0.0
        assert params.top_k == 0
        assert params.echo == False
        assert params.logprobs == None
        assert params.use_beam_search == False
        assert params.ignore_eos == False
        assert params.n == 1
        assert params.best_of == 1
        assert params.stop == None
        assert params.stream == False
        assert params.min_p == 0.0

    def test_custom_params(self):
        # Test initialization with custom values for VLLModelParams
        params = VLLModelParams(model="custom_model", max_tokens=1024)
        assert params.model == "custom_model"
        assert params.max_tokens == 1024

    def test_post_init(self):
        # Test post initialization
        params = VLLModelParams(model=None)
        assert params.model == "/opt/ds/model/deployed_model"


class TestMessage:
    def test_default_params(self):
        # Test the default initialization of Message
        params = Message()
        assert params.message_id == None
        assert params.parent_message_id == None
        assert params.session_id == None
        assert params.thread_id == None
        assert params.content == None
        assert params.payload == None
        assert params.status == Status.ACTIVE
        assert params.rate == MessageRate.DEFAULT
        assert params.role == None
        assert params.created == None
        assert params.answers == []
        assert params.model_params == VLLModelParams()

    def test_from_db_model(self):
        # Test creating a Message instance from a MessageModel object
        # Assuming a mock MessageModel object
        mock_date = datetime.now()
        mock_message_model = MagicMock(
            id=2,
            parent_id=1,
            playground_thread_id=3,
            content="test",
            payload={},
            model_params={
                "model": "test",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 1.0,
            },
            created=mock_date,
            status="active",
            rate=0,
            role="user",
        )

        message = Message.from_db_model(mock_message_model)
        assert message.message_id == 2
        assert message.parent_message_id == 1
        assert message.thread_id == 3
        assert message.content == "test"

        assert message.payload == {}
        assert message.model_params == VLLModelParams.from_dict(
            {"model": "test", "max_tokens": 2048, "temperature": 0.7, "top_p": 1.0}
        )
        assert message.created == mock_date
        assert message.status == "active"
        assert message.rate == 0
        assert message.role == "user"


class TestThread:
    def test_default_params(self):
        # Test the default initialization of Thread
        params = Thread()
        assert params.id == None
        assert params.name == None
        assert params.session_id == None
        assert params.created == None
        assert params.status == Status.ACTIVE
        assert params.messages == []

    def test_from_db_model(self):
        # Test creating a Thread instance from a ThreadModel object including associated messages
        # Assuming mock ThreadModel and MessageModel objects
        mock_date = datetime.now()
        mock_message_model_question = MagicMock(
            id=1,
            parent_id=None,
            playground_thread_id=3,
            content="question",
            payload={},
            model_params={
                "model": "test",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 1.0,
            },
            created=mock_date,
            status="active",
            rate=0,
            role=MessageRole.USER,
        )

        mock_message_model_answer = MagicMock(
            id=2,
            parent_id=1,
            playground_thread_id=3,
            content="answer",
            payload={},
            model_params={
                "model": "test",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 1.0,
            },
            created=mock_date,
            status="active",
            rate=0,
            role=MessageRole.SYSTEM,
        )

        mock_thread_model = MagicMock(
            id=1,
            name="test",
            playground_session_id=1,
            created=mock_date,
            status=Status.ACTIVE,
            messages=[mock_message_model_question, mock_message_model_answer],
        )

        thread = Thread.from_db_model(mock_thread_model, include_messages=True)
        assert thread.id == 1
        assert len(thread.messages) == 1
        assert thread.messages[0].message_id == 1
        assert thread.messages[0].content == "question"

        assert len(thread.messages[0].answers) == 1
        assert thread.messages[0].answers[0].message_id == 2
        assert thread.messages[0].answers[0].content == "answer"

        assert thread.session_id == 1
        assert thread.created == mock_date
        assert thread.status == Status.ACTIVE


class TestModelInfo:
    def test_initialization(self):
        # Test initialization of ModelInfo with specific model details
        model_info = ModelInfo(
            id="model1", name="Model One", endpoint="http://endpoint"
        )
        assert model_info.id == "model1"
        assert model_info.name == "Model One"
        assert model_info.endpoint == "http://endpoint"


class TestSession:
    def test_default_params(self):
        # Test the default initialization of Session
        params = Session()
        assert params.session_id == None
        assert params.created == None
        assert params.status == Status.ACTIVE
        assert params.threads == []
        assert params.model == ModelInfo()

    def test_from_db_model_with_threads(self):
        # Test creating a Session instance from a SessionModel object including associated threads
        # Assuming mock SessionModel and ThreadModel objects
        mock_date = datetime.now()
        mock_thread_model = MagicMock(
            id=1,
            name="test",
            session_id=1,
            created=mock_date,
            status=Status.ACTIVE,
            messages=[],
        )

        mock_session_model = MagicMock(
            id=1,
            name="test",
            created=mock_date,
            status=Status.ACTIVE,
            threads=[mock_thread_model],
            model_id="model_id",
            model_name="model_name",
            model_endpoint="model_endpoint",
        )

        session = Session.from_db_model(mock_session_model, include_threads=True)
        assert session.session_id == 1
        assert len(session.threads) == 1
        assert session.threads[0].id == 1
        assert session.created == mock_date
        assert session.model == ModelInfo(
            id="model_id", name="model_name", endpoint="model_endpoint"
        )
