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

from typing import Dict, List

from ads.aqua import logger
from ads.aqua.playground.const import MessageRole, Status
from ads.aqua.playground.db_context import ObjectType, db_context
from ads.aqua.playground.entities import (
    Message,
    SearchId,
    Session,
    Thread,
    VLLModelParams,
)
from ads.aqua.playground.errors import SessionNotFoundError
from ads.common.decorator import require_nonempty_arg
from ads.llm import ModelDeploymentVLLM
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


class ThreadApp:
    """
    Application class containing APIs for managing threads within Aqua Playground sessions.

    Methods
    -------

    get(thread_id: str)
        Retrieves a thread by its ID.
    deactivate(thread_id: str)
        Deactivates the thread with the given ID.
    activate(thread_id: str)
        Activates the thread with the given ID.
    post_message(thread_id: str, message: str, model_params: Dict = None)
        Posts a message to the specified thread.
    """

    @require_nonempty_arg("session_id", "The session ID must be provided.")
    def list(self, session_id: str, only_active: bool = True) -> List[Thread]:
        """
        Lists the registered playground session threads by session ID or model ID.

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
    def create(self, session_id: str, name: str, status: str = Status.ACTIVE) -> Thread:
        """
        Creates a new playground thread for the given session ID or model ID.

        Parameters
        ----------
        session_id: str
            The session ID to create the playground thread for.
            The model ID can be also provided. The session id will be retrieved by model ID.
        name: str
            The name of the thread.
        status: str
            The status of the thread. Can be either `active` or `archived`.

        Returns
        -------
        Thread
            The playground thread instance.
        """
        session = SessionApp().get(id=session_id, include_threads=False)
        return db_context.add_thread(
            session_id=session.session_id, name=name, status=status
        )

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

    @require_nonempty_arg(
        ["thread_id", "session_id"], "Either thread ID or session ID must be provided."
    )
    @require_nonempty_arg("message", "The message must be provided.")
    def post_message(
        self,
        message: str,
        thread_id: int = None,
        session_id: int = None,
        model_params: Dict = None,
    ) -> Thread:
        """
        Posts a message to the thread identified by the given ID.
        If session ID provided, then a new thread will be created.
        By default the model will not be invoked and the result will be empty.

        Parameters
        ----------
        message: str
            The content of the message to be posted.
        thread_id: str
            The ID of the thread where the message will be posted.
        session_id: str
            The ID of the session to where the thread will be created.
        model_params: (Dict, optional)
            Model parameters to be associated with the message.
            Currently supported VLLM+OpenAI parameters.

            --model-params '{
                "max_tokens":500,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.5,
                "model": "/opt/ds/model/deployed_model",
                ...}'

        Returns
        -------
        Thread
            The thread object containing only one question with answer.
        """
        model_params = VLLModelParams.from_dict(
            model_params or {}, ignore_unknown=True
        ).to_dict()

        # if thread needs to be created
        if not thread_id:
            thread = self.create(session_id=session_id, name=message[:50])
        else:
            # check if thread exists
            thread = self.get(thread_id=thread_id)

        # get the session info by thread ID
        session_obj = SessionApp().get(id=thread.session_id, include_threads=False)

        # register query message
        user_message = db_context.add_message_to_thread(
            thread_id=thread.id,
            content=message,
            role=MessageRole.USER,
            model_params=model_params,
            status=Status.ACTIVE,
        )

        # invoke the model
        model_deployment = ModelDeploymentVLLM(
            endpoint=f"{session_obj.model.endpoint.rstrip('/')}/predict",
            **model_params,
        )
        model_response_content = model_deployment(message)

        # register answer
        system_message = db_context.add_message_to_thread(
            thread_id=thread.id,
            parent_message_id=user_message.message_id,
            content=model_response_content,
            role=MessageRole.SYSTEM,
            model_params=model_params,
            status=Status.PENDING,
        )
        user_message.answers.append(system_message)

        thread.messages = [user_message]

        return thread


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
