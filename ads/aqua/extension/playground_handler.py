#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from dataclasses import dataclass, field
from typing import Dict

from tornado.web import HTTPError
import tornado
import random

from ads.aqua import logger
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.playground.entities import Message, Session, Thread
from ads.aqua.playground.model_invoker import ModelInvoker
from ads.aqua.playground.playground import MessageApp, SessionApp, ThreadApp
from ads.common.extended_enum import ExtendedEnumMeta
from ads.common.serializer import DataClassSerializable
from ads.common.utils import batch_convert_case


class Errors(str):
    INVALID_INPUT_DATA_FORMAT = "Invalid format of input data."
    NO_INPUT_DATA = "No input data provided."
    MISSING_REQUIRED_PARAMETER = "Missing required parameter: '{}'"


@dataclass
class NewSessionRequest(DataClassSerializable):
    """Dataclass representing the request on creating a new session."""

    model_id: str = None


@dataclass
class PostMessageRequest(DataClassSerializable):
    """Dataclass representing the request on posting a new message."""

    session: Session = field(default_factory=Session)
    thread: Thread = field(default_factory=Thread)
    message: Message = field(default_factory=Message)
    answer: Message = field(default_factory=Message)


class ChunkResponseStatus(str, metaclass=ExtendedEnumMeta):
    SUCCESS = "success"
    ERROR = "error"


@dataclass(repr=False)
class ChunkResponse(DataClassSerializable):
    """Class representing server response.

    Attributes
    ----------
    status: str
        Response status.
    message: (str, optional). Defaults to "".
        The response message.
    payload: (Dict, optional). Defaults  to None.
        The payload of the response.
    """

    status: str = None
    message: str = None
    payload: Dict = None


class AquaPlaygroundSessionHandler(AquaAPIhandler):
    """
    Handles the management and interaction with Playground sessions.

    Methods
    -------
    get(self, id="")
        Retrieves a list of sessions or a specific session by ID.
    post(self, *args, **kwargs)
        Creates a new playground session.
    read(self, id: str)
        Reads the detailed information of a specific Playground session.
    list(self)
        Lists all the playground sessions.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    def get(self, id=""):
        """
        Retrieve a list of all sessions or a specific session by its ID.

        Parameters
        ----------
        id: (str, optional)
            The ID of the session to retrieve. Defaults to an empty string,
            which implies fetching all sessions.

        Returns
        -------
        The session data or a list of sessions.
        """
        if not id:
            return self.list()
        return self.read(id)

    def read(self, id: str):
        """Read the information of a Playground session."""
        try:
            return self.finish(SessionApp().get(id=id, include_threads=True))
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def list(self):
        """List playground sessions."""
        try:
            return self.finish(SessionApp().list())
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def post(self, *args, **kwargs):
        """
        Creates a new Playground session by model ID.
        The session data is extracted from the JSON body of the request.
        If session for given model ID exists, then the existing session will be returned.

        Raises
        ------
        HTTPError
            If the input data is invalid or missing, or if an error occurs during session creation.
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT)

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        new_session_request = NewSessionRequest.from_dict(
            batch_convert_case(input_data, to_fmt="snake")
        )

        if not new_session_request.model_id:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("modelId"))

        try:
            self.finish(SessionApp().create(model_id=new_session_request.model_id))
        except Exception as ex:
            raise HTTPError(500, str(ex))


class AquaPlaygroundThreadHandler(AquaAPIhandler):
    """
    Handles the management and interaction with Playground threads.

    Methods
    -------
    get(self, thread_id="")
        Retrieves a list of threads or a specific thread by ID.
    post(self, *args, **kwargs)
        Creates a new playground thread.
    delete(self)
        Deletes (soft delete) a specified thread by ID.
    read(self, thread_id: str)
        Reads the detailed information of a specific Playground thread.
    list(self)
        Lists all the threads in a session.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    def get(self, thread_id: str = ""):
        """
        Retrieve a list of all threads or a specific thread by its ID.

        Parameters
        ----------
        thread_id (str, optional)
            The ID of the thread to retrieve. Defaults to an empty string,
            which implies fetching all threads.

        Returns
        -------
        The thread data or a list of threads.
        """
        if not thread_id:
            return self.list()
        return self.read(thread_id)

    def read(self, thread_id: str):
        """Read the information of a playground thread."""
        try:
            return self.finish(
                ThreadApp().get(thread_id=thread_id, include_messages=True)
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def list(self):
        """
        List playground threads.

        Args
        ----
        session_id: str
            The ID of the session to list associated threads.
        """
        session_id = self.get_argument("session_id")
        try:
            return self.finish(ThreadApp().list(session_id=session_id))
        except Exception as ex:
            raise HTTPError(500, str(ex))

    async def post(self, *args, **kwargs):
        """
        Adds a new message into the Playground thread.
        If the thread doesn't exist yet, then it will be created.
        """
        self.set_header("Content-Type", "application/json")
        self.set_header("Transfer-Encoding", "chunked")

        try:
            request_data: PostMessageRequest = PostMessageRequest.from_dict(
                batch_convert_case(self.get_json_body(), to_fmt="snake")
            )
        except Exception as ex:
            logger.debug(ex)
            error_msg = ChunkResponse(
                status=ChunkResponseStatus.ERROR,
                message=Errors.INVALID_INPUT_DATA_FORMAT,
            ).to_json()
            self.write(f"{len(error_msg):X}\r\n{error_msg}\r\n0\r\n\r\n")
            await self.flush()
            return

        thread_app = ThreadApp()
        message_app = MessageApp()
        # Register all entities in the DB
        try:
            # Add thread into DB if it not exists
            new_thread = thread_app.create(
                request_data.session.session_id,
                name=request_data.thread.name,
                thread_id=request_data.thread.id,
            )

            # Add user message into DB
            new_user_message = message_app.create(
                thread_id=new_thread.id,
                content=request_data.message.content,
                message_id=request_data.message.message_id,
                parent_message_id=request_data.message.parent_message_id,
                role=request_data.message.role,
                rate=request_data.message.rate,
                payload=request_data.message.payload,
                model_params=request_data.message.model_params.to_dict(),
            )

            # Add system answer into DB
            new_system_message = message_app.create(
                thread_id=new_thread.id,
                content=request_data.answer.content,
                message_id=request_data.answer.message_id,
                parent_message_id=request_data.answer.parent_message_id,
                role=request_data.answer.role,
                rate=request_data.answer.rate,
                payload=request_data.answer.payload,
                model_params=request_data.answer.model_params.to_dict(),
            )

            # Send initial OK status to the client
            initial_response = ChunkResponse(
                status=ChunkResponseStatus.SUCCESS, message=""
            ).to_json()

            self.write(f"{len(initial_response):X}\r\n{initial_response}\r\n")
            await self.flush()
        except Exception as ex:
            logger.debug(ex)
            error_msg = ChunkResponse(
                status=ChunkResponseStatus.ERROR, message=str(ex)
            ).to_json()
            self.write(f"{len(error_msg):X}\r\n{error_msg}\r\n0\r\n\r\n")
            await self.flush()
            return

        try:
            model_response_text = ""
            model_invoker = ModelInvoker(
                endpoint=f"{request_data.session.model.endpoint.rstrip('/')}/predict",
                prompt=request_data.message.content,
                params=request_data.message.model_params.to_dict(),
            )
            for item in model_invoker.invoke():
                if item.startswith("data"):
                    if "[DONE]" in item:
                        continue
                    item_json = json.loads(item[6:])
                else:
                    item_json = json.loads(item)

                if item_json.get("object") == "error":
                    # {"object":"error","message":"top_k must be -1 (disable), or at least 1, got 0.","type":"invalid_request_error","param":null,"code":null}
                    raise HTTPError(400, item_json.get("message"))
                else:
                    chunk = ChunkResponse(
                        status=ChunkResponseStatus.SUCCESS,
                        message="",
                        payload=item_json["choices"][0]["text"],
                    ).to_json()

                    model_response_text += item_json["choices"][0]["text"]

                    # update system message in DB
                    message_app.update(
                        message_id=new_system_message.message_id,
                        content=model_response_text,
                        rate=new_system_message.rate,
                        status=new_system_message.status,
                    )

                    self.write(f"{len(chunk):X}\r\n{chunk}\r\n")
                    await self.flush()
                    await tornado.gen.sleep(random.random() * 0.01 + 0.05)

            # Indicate the end of the response
            self.write("0\r\n\r\n")
            await self.flush()
        except Exception as ex:
            logger.debug(ex)
            # Handle unexpected errors
            error_msg = ChunkResponse(
                status=ChunkResponseStatus.ERROR, message=str(ex)
            ).to_json()
            self.write(f"{len(error_msg):X}\r\n{error_msg}\r\n0\r\n\r\n")
            await self.flush()

    def delete(self, *args, **kwargs):
        """
        Deletes (soft delete) the thread by ID.

        Args
        ----
        thread_id: str
            The ID of the thread to be deleted.
        """
        thread_id = self.get_argument("threadId")
        if not thread_id:
            raise HTTPError(
                400, Errors.Errors.MISSING_REQUIRED_PARAMETER.format("threadId")
            )

        # Only soft deleting with updating a status field.
        try:
            ThreadApp().deactivate(thread_id=thread_id)
            self.set_status(204)  # no content
            self.finish()
        except Exception as ex:
            raise HTTPError(500, str(ex))


__handlers__ = [
    ("playground/session/?([^/]*)", AquaPlaygroundSessionHandler),
    ("playground/thread/?([^/]*)", AquaPlaygroundThreadHandler),
]
