#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import datetime
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ads.aqua.playground.const import MessageRate, Status, MessageRole
from ads.aqua.playground.db_models import MessageModel, SessionModel, ThreadModel
from ads.common.serializer import DataClassSerializable


@dataclass
class SearchId:
    """
    Class helper to encode a search id.
    """

    model_id: str = None
    record_id: str = None

    @classmethod
    def parse(cls, id: str) -> "SearchId":
        """
        The method differentiates between record ID and model ID based on the content of the ID.
        If the ID contains 'ocid', it is treated as model ID.

        Parameters
        ----------
        id: str
            Input ID to parse.
        """
        result = cls()

        if not id:
            raise ValueError(
                "Incorrect id was provided. "
                "It should either be a model ID or a record ID."
            )

        if "ocid" in (id or ""):
            result.model_id = id
        else:
            result.record_id = id

        return result


@dataclass(repr=False)
class VLLModelParams(DataClassSerializable):
    """
    Parameters specific to Versatile Large Language Model.

    Attributes
    ----------
    model: (str, optional)
        Model name.
    max_tokens: (int, optional)
        Maximum number of tokens to generate.
    temperature: (float, optional)
        Controls randomness in generation.
    top_p: (float, optional)
        Top probability mass.
    frequency_penalty: (float, optional)
        Penalizes new tokens based on their existing frequency.
    presence_penalty: (float, optional)
        Penalizes new tokens based on their presence.
    top_k: (int, optional)
        Keeps only top k candidates at each generation step.
    echo: (bool, optional)
        Echoes the input text in the output.
    logprobs: (int, optional)
        Number of log probabilities to return.
    use_beam_search: (bool, optional)
        Whether to use beam search for generation.
    ignore_eos: (bool, optional)
        Whether to ignore end-of-sequence tokens during generation.
    n: (int, optional)
        Number of output sequences to return for a given prompt.
    best_of: (int, optional)
        Controls how many completions to generate for each prompt.
    stop: (List[str], optional)
        Stop words or phrases to use when generating.
    stream: (bool, optional)
        Indicates whether the response should be streamed.
    min_p: (float, optional)
        Minimum probability threshold for token selection.
    """

    model: Optional[str] = "/opt/ds/model/deployed_model"
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    top_k: Optional[int] = 1
    echo: Optional[bool] = False
    logprobs: Optional[int] = None
    use_beam_search: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    stop: Optional[List[str]] = field(default_factory=list)
    stream: Optional[bool] = True
    min_p: Optional[float] = 0.0

    def __post_init__(self):
        if not self.model:
            self.model = "/opt/ds/model/deployed_model"
        if self.stream is None:
            self.stream = True


@dataclass(repr=False)
class Message(DataClassSerializable):
    """
    Data class representing a message in a thread.

    Attributes
    ----------
    id: str
        Unique identifier of the message.
    parent_id: int
        Identifier of the parent message.
    thread_id: int
        Identifier of the thread to which the message belongs.
    session_id: int
        Identifier of the session to which the message thread belongs.
    content: str
        The actual content of the message.
    payload: Dict
        Additional payload associated with the message.
    status: str
        Status of the message. Can be `active` or `archived`.
    rate: int
        Rating of the message, based on the MessageRate enum.
    role: str
        Role of the message, based on the MessageRole enum.
    created: datetime.datetime
        Creation timestamp of the message.
    updated: datetime.datetime
        Updating timestamp of the message.
    answers: List[Message]
        List of system messages for the user message.
    """

    message_id: str = None
    parent_message_id: str = None
    session_id: str = None
    thread_id: str = None
    content: str = None
    payload: Dict = None
    model_params: VLLModelParams = field(default_factory=VLLModelParams)
    status: str = Status.ACTIVE
    rate: int = MessageRate.DEFAULT
    role: str = None
    created: datetime.datetime = None
    updated: datetime.datetime = None
    answers: List["Message"] = field(default_factory=list)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.status:
            self.status = Status.ACTIVE
        if not self.rate:
            self.rate = MessageRate.DEFAULT
        if not self.role:
            self.role = MessageRole.USER

    @classmethod
    def from_db_model(cls, data: MessageModel) -> "Message":
        """
        Creates Message instance from MessageModel object.

        Parameters
        ----------
        data: MessageModel
            The DB representation of the message.

        Returns
        -------
        Message

             The instance of the playground message.
        """
        return cls(
            message_id=data.id,
            parent_message_id=data.parent_id,
            content=data.content,
            thread_id=data.playground_thread_id,
            session_id=data.thread.playground_session_id,
            created=data.created,
            updated=data.updated,
            status=data.status,
            rate=data.rate,
            role=data.role,
            payload=data.payload,
            model_params=VLLModelParams.from_dict(data.model_params),
        )


@dataclass(repr=False)
class Thread(DataClassSerializable):
    """
    Data class representing a thread in a session.

    Attributes
    ----------
    id: str
        Unique identifier of the thread.
    name: str
        Name of the thread.
    session_id: str
        Identifier of the session to which the thread belongs.
    created: datetime.datetime
        Creation timestamp of the thread.
    updated: datetime.datetime
        Updating timestamp of the thread.
    status: str
        Status of the message. Can be `active` or `archived`.
    messages: List[Message]
        List of messages in the thread.
    """

    id: str = None
    name: str = None
    session_id: str = None
    created: datetime.datetime = None
    updated: datetime.datetime = None
    status: str = Status.ACTIVE
    messages: List[Message] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.status:
            self.status = Status.ACTIVE

    @classmethod
    def from_db_model(
        cls, data: ThreadModel, include_messages: bool = False
    ) -> "Thread":
        """
        Creates Thread instance from ThreadModel object.

        Parameters
        ----------
        data: ThreadModel
            The DB representation of the thread.
        include_messages: (bool, optional)
            Include the associated messages into the result.

        Returns
        -------
        Thread
            The instance of the playground thread.
        """
        obj = cls(
            id=data.id,
            name=data.name,
            session_id=data.playground_session_id,
            created=data.created,
            updated=data.updated,
            status=data.status,
        )

        if include_messages and data.messages:
            # Assign the list of answers to the parent messages
            messages = [
                Message.from_db_model(data=message_model)
                for message_model in data.messages
            ]

            # Filter and return only root messages (those with parent_id == None)
            result_messages = [msg for msg in messages if not msg.parent_message_id]

            # Group messages by parent ID
            message_map = defaultdict(list)
            for msg in messages:
                message_map[msg.parent_message_id].append(msg)

            # Add child messages
            for msg in result_messages:
                msg.answers = message_map[msg.message_id]

            obj.messages = result_messages

        return obj


@dataclass(repr=False)
class ModelInfo(DataClassSerializable):
    """
    Data class representing model deployment details.

    Attributes
    ----------
    id: (str, optional)
        The model deployment ID.
    name: (str, optional)
        The model deployment name.
    endpoint: (str, optional)
        The model deployment endpoint.
    """

    id: Optional[str] = None
    name: Optional[str] = None
    endpoint: Optional[str] = None


@dataclass(repr=False)
class Session(DataClassSerializable):

    """
    Data class representing a session in the Aqua Playground.

    Attributes
    ----------
    id: str
        Unique identifier of the session.
    created: datetime.datetime
        Creation timestamp of the session.
    updated: datetime.datetime
        Updating timestamp of the session.
    status: str
        Status of the message. Can be `active` or `archived`.
    threads: List[Thread]
        List of threads in the session.
    model: ModelInfo
        Model deployment details.
    """

    session_id: str = None
    created: datetime.datetime = None
    updated: datetime.datetime = None
    status: str = Status.ACTIVE
    threads: List[Thread] = field(default_factory=list)
    model: ModelInfo = field(default_factory=ModelInfo)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if not self.status:
            self.status = Status.ACTIVE

    @classmethod
    def from_db_model(
        cls, data: SessionModel, include_threads: bool = False
    ) -> "Session":
        """
        Creates Session instance form SessionModel object.

        Parameters
        ----------
        data: SessionModel
            The DB representation of the session.
        include_threads: (bool, optional)
            Whether to include the threads into the result.

        Returns
        -------
        Session
            The instance of the playground session.
        """

        obj = cls(
            session_id=data.id,
            created=data.created,
            updated=data.updated,
            status=data.status,
            model=ModelInfo(
                id=data.model_id, name=data.model_name, endpoint=data.model_endpoint
            ),
        )

        if include_threads and data.threads:
            obj.threads = [
                Thread.from_db_model(thread_model) for thread_model in data.threads
            ]

        return obj
