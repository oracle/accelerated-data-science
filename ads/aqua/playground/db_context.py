#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from ads.aqua.playground.const import MessageRate, MessageRole, ObjectType, Status
from ads.aqua.playground.db_models import Base, MessageModel, SessionModel, ThreadModel
from ads.aqua.playground.entities import Message, Session, Thread
from ads.aqua.playground.errors import (
    MessageNotFoundError,
    SessionNotFoundError,
    ThreadNotFoundError,
)

DATABASE_NAME = "playground.db"
DATABASE_PATH = os.environ.get("AQUA_PLAYGROUND") or os.path.join(
    os.path.abspath(os.path.expanduser("~")), ".aqua"
)


OBJECT_MODEL_MAP = {
    ObjectType.SESSION: SessionModel,
    ObjectType.THREAD: ThreadModel,
    ObjectType.MESSAGE: MessageModel,
}


class DBContext:
    """
    A class to handle database operations for Playground sessions, threads, and messages.

    Attributes
    ----------
    engine (Engine): SQLAlchemy engine instance for database connection.
    DBSession (sessionmaker): Factory for creating new SQLAlchemy session objects.
    """

    def __init__(self, db_url: str, echo: bool = False):
        """
        Initializes the database context with a given database URL.

        Parameters
        ----------
        db_url (str): A database URL that indicates database dialect and connection arguments.
        echo: (bool, optional). Whether to show the debug information or not.
        """
        self.engine = create_engine(db_url, echo=echo, future=True)
        self.DBSession = scoped_session(
            sessionmaker(bind=self.engine, future=True, expire_on_commit=True)
        )
        self.init_db()

    def init_db(self):
        Base.metadata.create_all(self.engine)

    def get_sessions(
        self, only_active: bool = True, include_threads: bool = False
    ) -> List[Session]:
        """
        Retrieves all threads for a specific playground db_session.

        Parameters
        ----------
        only_active: (bool, optional). Defaults to True.
            Whether to load all or only active sessions.
        include_threads: (bool, optional). Defaults to False.
            Whether to include the associated threads or not.

        Returns
        -------
        List[SessionModel]
            A list of playground sessions.
        """
        with self.DBSession() as db_session:
            db_session.expire_on_commit
            query = db_session.query(SessionModel)

            if only_active:
                query.filter_by(status=Status.ACTIVE)

            return [
                Session.from_db_model(session_model, include_threads=include_threads)
                for session_model in query.all()
            ]

    def get_session(
        self,
        session_id: str = None,
        model_id: str = None,
        include_threads: bool = False,
    ) -> Optional[Session]:
        """
        Retrieves a playground session by its session ID or model ID.

        Parameters
        ----------
        session_id: (str, optional)
            The unique session identifier for the playground db_session.
        model_id: (str, optional)
            The unique model identifier for the playground db_session.
        include_threads: (bool, optional). Defaults to False.
            Whether to include the associated threads or not.

        Returns
        -------
        Optional[SessionModel]
            The retrieved playground session if found, else None.

        Raises
        ------
        ValueError
            If neither session_id nor model_id was provided.
        SessionNotFoundError
            If session with the provided ID doesn't exist.
        """

        if not (session_id or model_id):
            raise ValueError("Either session ID or model ID need to be provided.")

        with self.DBSession() as db_session:
            query = db_session.query(SessionModel)
            if session_id:
                session_model = query.filter_by(id=session_id).first()
            else:
                session_model = query.filter_by(model_id=model_id).first()

            if not session_model:
                raise SessionNotFoundError(session_id or model_id)

            return Session.from_db_model(session_model, include_threads=include_threads)

    def add_session(
        self,
        model_id: str,
        model_name: str,
        model_endpoint: str,
        session_id: str = None,
        status: str = Status.ACTIVE,
    ) -> Session:
        """
        Adds a new playground session to the database.

        Parameters
        ----------
        model_id (str): The unique model identifier for the new db_session.
        model_name (str): The name of the model.
        model_endpoint (str): The model endpoint.
        session_id (str, optional): The ID of the session.
        status (str, optional): The status of the db_session.

        Returns
        -------
        SessionModel
            The newly created playground db_session.
        """
        with self.DBSession() as db_session:
            new_session = SessionModel(
                id=session_id,
                model_id=model_id,
                model_name=model_name,
                model_endpoint=model_endpoint,
                created=datetime.now(),
                updated=datetime.now(),
                status=status,
            )
            db_session.add(new_session)
            db_session.commit()

            return Session.from_db_model(new_session)

    def get_session_threads(
        self,
        session_id: str,
        only_active: bool = True,
    ) -> List[Thread]:
        """
        Retrieves all threads for a specific playground db_session.

        Parameters
        ----------
        session_id: (str)
            The ID of the session for which to retrieve threads.
        only_active: (bool, optional). Defaults to True.
            Whether to load all or only active sessions.

        Returns
        -------
        List[ThreadModel]
            A list of playground threads associated with the db_session.
        """
        with self.DBSession() as db_session:
            query = db_session.query(ThreadModel).filter_by(
                playground_session_id=session_id
            )

            if only_active:
                query.filter_by(status=Status.ACTIVE)

            return [Thread.from_db_model(thread_model) for thread_model in query.all()]

    def get_thread(
        self,
        thread_id: str = None,
        include_messages: bool = False,
    ) -> Optional[Thread]:
        """
        Retrieves a playground thread by its ID.

        Parameters
        ----------
        thread_id: str
            The unique thread identifier.
        include_messages: (bool, optional). Defaults to False.
            Whether to include the associated messages or not.

        Returns
        -------
        Optional[ThreadModel]
            The retrieved playground thread if found, else None.

        Raises
        ------
        ThreadNotFoundError
            If thread with provided ID doesn't exist.
        """
        with self.DBSession() as db_session:
            thread_model = db_session.query(ThreadModel).filter_by(id=thread_id).first()

            if not thread_model:
                raise ThreadNotFoundError(thread_id=thread_id)

            return Thread.from_db_model(thread_model, include_messages=include_messages)

    def add_thread(
        self,
        session_id: str,
        name: str,
        thread_id: str = None,
        status: str = Status.ACTIVE,
    ) -> Thread:
        """
        Adds a new thread to an existing playground db_session.

        Parameters
        ----------
        session_id (str): The ID of the session to which the thread belongs.
        name (str): The name of the thread.
        thread_id (str, optional): The thread ID.
        status (str, optional): The status of the thread. Defaults to active.

        Returns
        -------
        Thread
            The newly created playground thread.
        """
        with self.DBSession() as db_session:
            new_thread = ThreadModel(
                id=thread_id,
                playground_session_id=session_id,
                name=name,
                created=datetime.now(),
                updated=datetime.now(),
                status=status,
            )
            db_session.add(new_thread)
            db_session.commit()

            return Thread.from_db_model(new_thread)

    def update_thread(
        self, thread_id: str, name: str = None, status: str = None
    ) -> Optional[Thread]:
        """
        Updates a playground thread by its ID.

        Parameters
        ----------
        thread_id: str
            The unique thread identifier.

        Returns
        -------
        Optional[Message]
            Updated thread or None.

        Raises
        ------
        ThreadNotFoundError
            If message with provided ID doesn't exist.
        """
        with self.DBSession() as db_session:
            thread_model = db_session.query(ThreadModel).filter_by(id=thread_id).first()

            if not thread_model:
                raise ThreadNotFoundError(thread_id=thread_id)

            thread_model.name = name or thread_model.name
            thread_model.status = status or thread_model.status
            thread_model.updated = datetime.now()

            db_session.commit()
            return Thread.from_db_model(thread_model)

    def get_thread_messages(
        self, thread_id: str, only_active: bool = True
    ) -> List[Message]:
        """
        Retrieves all messages in a specific playground thread.

        Parameters
        ----------
        thread_id (str): The ID of the thread for which to retrieve messages.
        only_active: (bool, optional). Defaults to True.
            Whether to load all or only active messages.

        Returns
        -------
        List[Message]
            A list of playground messages in the thread.
        """
        with self.DBSession() as db_session:
            query = db_session.query(MessageModel).filter_by(
                playground_thread_id=thread_id
            )

            if only_active:
                query.filter_by(status=Status.ACTIVE)

            return [
                Message.from_db_model(message_model) for message_model in query.all()
            ]

    def add_message_to_thread(
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
        Adds a message to a specific playground thread.

        Parameters
        ----------
        thread_id (str): The ID of the thread to which the message will be added.
        content (str): The text content of the message.
        message_id (str, optional): The message ID.
        parent_message_id (str, optional): The parent message.
        payload (Dict, optional): The model payload.
        model_params (Dict, optional): The model parameters.
        status (str): The status of the message.
        role (str): The role of the message (e.g., 'user', 'system').

        Returns
        -------
        Message
            The newly created playground message.
        """
        with self.DBSession() as db_session:
            new_message = MessageModel(
                id=message_id,
                playground_thread_id=thread_id,
                parent_id=parent_message_id,
                content=content or "",
                created=datetime.now(),
                updated=datetime.now(),
                status=status or Status.ACTIVE,
                role=role or MessageRole.USER,
                rate=rate or MessageRate.DEFAULT,
                payload=payload or {},
                model_params=model_params or {},
            )
            db_session.add(new_message)
            db_session.commit()
            return Message.from_db_model(new_message)

    def get_message(
        self,
        message_id: str = None,
    ) -> Optional[Message]:
        """
        Retrieves a playground message by its ID.

        Parameters
        ----------
        message_id: str
            The unique message identifier.

        Returns
        -------
        Optional[Message]
            The retrieved playground message if found, else None.

        Raises
        ------
        MessageNotFoundError
            If message with provided ID doesn't exist.
        """
        with self.DBSession() as db_session:
            message_model = (
                db_session.query(MessageModel).filter_by(id=message_id).first()
            )

            if not message_model:
                raise MessageNotFoundError(message_id == message_id)

            return Message.from_db_model(message_model)

    def update_message(
        self, message_id: str, content: str = None, status: str = None, rate: int = None
    ) -> Optional[Message]:
        """
        Updates a playground message by its ID.

        Parameters
        ----------
        message_id: str
            The unique message identifier.

        Returns
        -------
        Optional[Message]
            The retrieved playground message if found, else None.

        Raises
        ------
        MessageNotFoundError
            If message with provided ID doesn't exist.
        """
        with self.DBSession() as db_session:
            message_model = (
                db_session.query(MessageModel).filter_by(id=message_id).first()
            )

            if not message_model:
                raise MessageNotFoundError(message_id=message_id)

            message_model.content = content or message_model.content
            message_model.status = status or message_model.status
            message_model.rate = rate or message_model.rate
            message_model.updated = datetime.now()

            db_session.commit()
            return Message.from_db_model(message_model)

    def update_status(self, object_type: str, object_id: str, status: str):
        """
        Update the status of a session, thread, or message.

        Parameters
        ----------
        object_type (str): The type of object to update ('session', 'thread', or 'message').
        object_id (str): The ID of the object to update.
        status (str): The new status to set for the object.
        """
        with self.DBSession() as db_session:
            if object_type in OBJECT_MODEL_MAP:
                db_session.query(OBJECT_MODEL_MAP[object_type]).filter_by(
                    id=object_id
                ).update({"status": status, "updated": datetime.now()})
                db_session.commit()

    def delete_object(self, object_type: str, object_id: str):
        """
        Delete a session, thread, or message from the database.

        Parameters
        ----------
        object_type (str): The type of object to delete ('session', 'thread', or 'message').
        object_id (str): The ID of the object to delete.
        """
        with self.DBSession() as db_session:
            if object_type in OBJECT_MODEL_MAP:
                db_session.query(OBJECT_MODEL_MAP[object_type]).filter_by(
                    id=object_id
                ).delete()
                db_session.commit()


###################### INIT DB CONTEXT######################################
os.makedirs(DATABASE_PATH, exist_ok=True)
db_context = DBContext(db_url=f"sqlite:///{DATABASE_PATH}/{DATABASE_NAME}")
############################################################################
