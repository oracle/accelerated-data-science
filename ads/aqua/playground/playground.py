#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
This Python module is part of Oracle's Artificial Intelligence QA Playground,
a tool for managing and interacting with AI Quick Actions models. It includes classes
for handling sessions, threads, messages, and model parameters, along with custom exceptions
and utilities for session and thread management.
"""

from typing import Dict, Generator, List

from ads.aqua import logger
from ads.aqua.playground.const import MessageRate, MessageRole, ObjectType, Status
from ads.aqua.playground.db_context import ObjectType, db_context
from ads.aqua.playground.entities import (
    Message,
    SearchId,
    Session,
    Thread,
    VLLModelParams,
)
from ads.aqua.playground.errors import SessionNotFoundError, ThreadNotFoundError
from ads.aqua.playground.model_invoker import ModelInvoker
from ads.common.decorator import require_nonempty_arg
from ads.model.deployment.model_deployment import ModelDeployment


class SessionApp:
    """
    Application class containing APIs for managing Aqua Playground sessions.


    Methods
    -------
    list(self, only_active: bool = True, include_threads: bool = False) -> Session
        Lists the registered playground sessions.
    get(model_id: str) -> Session
        Retrieves a session associated with the specified model ID.
    deactivate(model_id: str)
        Deactivates the session associated with the specified model ID.
    activate(model_id: str)
        Activates the session associated with the specified model ID.
    """

    def list(
        self, only_active: bool = True, include_threads: bool = False
    ) -> List[Session]:
        """
        Lists the registered playground sessions.

        Parameters
        ----------
        only_active: (bool, optional). Defaults to True.
            Whether to load all or only active sessions.
        include_threads: (bool, optional). Defaults to False.
            Whether to include the associated threads or not.

        Returns
        -------
        List[Session]
            The list of playground sessions.
        """
        return db_context.get_sessions(
            only_active=only_active, include_threads=include_threads
        )

    @require_nonempty_arg("id", "Either session ID or model ID need to be provided.")
    def get(self, id: str, include_threads: bool = False) -> Session:
        """
        Retrieve a playground session details by its session ID or model ID.

        The method differentiates between session ID and model ID based on the content of the ID.
        If the ID contains 'ocid', it is treated as model ID.

        Parameters
        ----------
        id: str
            The session ID or model ID of the playground session.
        include_threads: (bool, optional)
            Whether include threads in result or not.

        Returns
        -------
        Optional[Session]
            The retrieved playground session if found, else None.
        """
        search_id = SearchId.parse(id)
        return db_context.get_session(
            session_id=search_id.record_id,
            model_id=search_id.model_id,
            include_threads=include_threads,
        )

    @require_nonempty_arg("model_id", "The model ID must be provided.")
    def create(self, model_id: str) -> Session:
        """
        Creates a new playground session for the given model ID.
        If the session with the given model ID already exists, then it will be returned.

        Parameters
        ----------
        model_id: str
            The model ID to create the playground session for.

        Returns
        -------
        Session
            The playground session instance.

        Raises
        ------
        ValueError
            If model ID not provided.
        """

        try:
            session = self.get(id=model_id, include_threads=True)
            logger.info(
                "A Session with the provided model ID already exists. "
                "Returning the existing session."
            )
        except SessionNotFoundError:
            model_deployment = ModelDeployment.from_id(model_id)
            session = db_context.add_session(
                model_id=model_deployment.model_deployment_id,
                model_name=model_deployment.display_name,
                model_endpoint=model_deployment.url,
            )

        return session

    @require_nonempty_arg("session_id", "The session ID must be provided.")
    def activate(self, session_id: str):
        """
        Activates the session associated with the given ID.

        Parameters
        ----------
        session: str
            The ID of the playground session to deactivate.

        Raises
        ------
        ValueError
            If session ID not provided.
        """
        db_context.update_status(
            object_type=ObjectType.SESSION, object_id=session_id, status=Status.ACTIVE
        )

    @require_nonempty_arg("session_id", "The session ID must be provided.")
    def deactivate(self, session_id: str):
        """
        Deactivates the session associated with the given ID.

        Parameters
        ----------
        session: str
            The ID of the playground session to deactivate.
        """
        db_context.update_status(
            object_type=ObjectType.SESSION, object_id=session_id, status=Status.ARCHIVED
        )

    @require_nonempty_arg("prompt", "The message must be provided.")
    @require_nonempty_arg("endpoint", "The model endpoint must be provided.")
    def invoke_model(
        self,
        endpoint: str,
        prompt: str,
        params: Dict = None,
    ) -> Generator[str, None, None]:
        """
        Generator to invoke the model and streams the result.

        Parameters
        ----------
        endpoint:str
            The URL endpoint to send the request.
        prompt: str
            The content of the message to be posted.
        params: (Dict, optional)
            Model parameters to be associated with the message.
            Currently supported VLLM+OpenAI parameters.

            --model-params '{
                "max_tokens":500,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.5,
                "model": "/opt/ds/model/deployed_model",
                ...}'

        Yields:
        str
            A line of the streamed response.
        """
        yield from ModelInvoker(
            endpoint=endpoint,
            prompt=prompt,
            params=VLLModelParams.from_dict(params).to_dict(),
        ).invoke()


class ThreadApp:
    """
    Application class containing APIs for managing threads within Aqua Playground sessions.

    Methods
    -------
    list(self, session_id: str, only_active: bool = True) -> List[Thread]
        Lists the registered playground session threads by session ID.
    get(thread_id: str)
        Retrieves a thread by its ID.
    create(self, session_id: str, name: str, thread_id: str = None, status: str = Status.ACTIVE) -> Thread
        Creates a new playground thread for the given session ID.
    deactivate(thread_id: str)
        Deactivates the thread with the given ID.
    activate(thread_id: str)
        Activates the thread with the given ID.
    """

    @require_nonempty_arg("session_id", "The session ID must be provided.")
    def list(self, session_id: str, only_active: bool = True) -> List[Thread]:
        """
        Lists the registered playground threads by session ID.

        Parameters
        ----------
        session_id: str
            The session ID to get the playground threads for.
            The model ID can be also provided. The session id will be retrieved by model ID.
        only_active: (bool, optional). Defaults to True.
            Whether to load all or only active threads.

        Returns
        -------
        List[Thread]
            The list of playground session threads.
        """

        session = SessionApp().get(id=session_id, include_threads=False)
        return db_context.get_session_threads(
            session_id=session.session_id, only_active=only_active
        )

    @require_nonempty_arg("thread_id", "The thread ID must be provided.")
    def get(self, thread_id: str, include_messages: bool = False) -> Thread:
        """
        Retrieve a thread based on its ID.

        Parameters
        ----------
        thread_id: str
            The ID of the thread to be retrieved.
        include_messages: (bool, optional). Defaults to False.
            Whether include messages in result or not.

        Returns
        -------
        Thread
            The playground session thread.

        Raise
        -----
        ThreadNotFoundError
            If thread doesn't exist.
        """
        return db_context.get_thread(
            thread_id=thread_id, include_messages=include_messages
        )

    @require_nonempty_arg("session_id", "The session ID must be provided.")
    @require_nonempty_arg("name", "The name for the new thread must be provided.")
    def create(
        self,
        session_id: str,
        name: str,
        thread_id: str = None,
        status: str = Status.ACTIVE,
    ) -> Thread:
        """
        Creates a new playground thread for the given session ID or model ID.

        Parameters
        ----------
        session_id: str
            The session ID to create the playground thread for.
            The model ID can be also provided. The session id will be retrieved by model ID.
        name: str
            The name of the thread.
        thread_id: (str, optional)
            The ID of the thread. Will be auto generated if not provided.
        status: (str, optional)
            The status of the thread. Can be either `active` or `archived`.

        Returns
        -------
        Thread
            The playground thread instance.
        """
        session = SessionApp().get(id=session_id, include_threads=False)
        thread = None
        if thread_id:
            try:
                thread = db_context.update_thread(
                    thread_id=thread_id, name=name, status=status
                )
            except ThreadNotFoundError:
                pass

        if not thread:
            thread = db_context.add_thread(
                session_id=session.session_id,
                name=name,
                status=status,
                thread_id=thread_id,
            )

        return thread

    @require_nonempty_arg("thread_id", "The thread ID must be provided.")
    def deactivate(self, thread_id: str):
        """
        Deactivates the thread with the specified ID.

        Parameters
        ----------
        thread_id: str
            The ID of the thread to be deactivated.
        """
        db_context.update_status(
            object_type=ObjectType.THREAD, object_id=thread_id, status=Status.ARCHIVED
        )

    @require_nonempty_arg("thread_id", "The thread ID must be provided.")
    def activate(self, thread_id: str):
        """
        Activates the thread with the specified ID.

        Parameters
        ----------
        thread_id: str
            The ID of the thread to be activated.
        """
        db_context.update_status(
            object_type=ObjectType.THREAD, object_id=thread_id, status=Status.ACTIVE
        )


class MessageApp:
    """
    Application class containing APIs for managing messages within Aqua Playground thread.

    Methods
    -------

    create(self, thread_id: str, content: str, ...) -> Message
        Posts a message to the specified thread.
    """

    @require_nonempty_arg("thread_id", "The session ID must be provided.")
    def create(
        self,
        thread_id: str,
        content: str,
        message_id: str = None,
        parent_message_id: str = None,
        role: str = MessageRole.USER,
        rate: int = MessageRate.DEFAULT,
        payload: Dict = None,
        model_params: Dict = None,
        status: str = Status.ACTIVE,
    ) -> Message:
        """
        Creates a new message for the given thread ID.

        Parameters
        ----------
        thread_id: str
            The ID of the thread to which the message will be added.
        content: str
            The text content of the message.
        message_id: (str, optional)
            The message ID.
        parent_message_id: (str, optional)
            The parent message.
        payload: (Dict, optional)
            The model payload.
        model_params: (Dict, optional)
            The model parameters.
        status: (str)
            The status of the message.
        role: (str)
            The role of the message (e.g., 'user', 'system').

        Returns
        -------
        Message
            The playground message instance.
        """
        return db_context.add_message_to_thread(
            thread_id=thread_id,
            content=content,
            message_id=message_id,
            parent_message_id=parent_message_id,
            role=role,
            rate=rate,
            payload=payload,
            model_params=model_params,
            status=status,
        )

    @require_nonempty_arg("message_id", "The message ID must be provided.")
    def update(
        self,
        message_id: str,
        content: str,
        rate: int = MessageRate.DEFAULT,
        status: str = Status.ACTIVE,
    ) -> Message:
        """
        Updates a message by provided ID.

        Parameters
        ----------
        thread_id: str
            The ID of the thread to which the message will be added.
        content: str
            The text content of the message.
        message_id: (str, optional)
            The message ID.
        parent_message_id: (str, optional)
            The parent message.
        payload: (Dict, optional)
            The model payload.
        model_params: (Dict, optional)
            The model parameters.
        status: (str)
            The status of the message.
        role: (str)
            The role of the message (e.g., 'user', 'system').

        Returns
        -------
        Message
            The playground message instance.
        """
        return db_context.update_message(
            content=content,
            message_id=message_id,
            rate=rate,
            status=status,
        )


class PlaygroundApp:
    """
    Aqua Playground Application.

    Attributes
    ----------
    session: SessionApp
        Managing playground sessions.
    thread: ThreadApp
        Managing playground threads within sessions.
    """

    session = SessionApp()
    thread = ThreadApp()
    message = MessageApp()
