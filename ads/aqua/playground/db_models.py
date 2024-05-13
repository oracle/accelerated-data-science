#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from datetime import datetime
from typing import List
import uuid

from sqlalchemy import JSON, TIMESTAMP, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    """
    Represents a playground session table in the database.


    Attributes:
        id (Mapped[str]): The primary key, the UUID.
        model_id (Mapped[str]): The id of the model.
        model_name: (Mapped[str]): The name of the model.
        model_endpoint: (Mapped[str]): The model endpoint.
        created (Mapped[datetime]): The creating timestamp of the session.
        updated (Mapped[datetime]): The updating timestamp of the session.
        status (Mapped[str]): The status of the session.
    """

    __tablename__ = "playground_session"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(String, unique=True)
    model_name: Mapped[str] = mapped_column(String)
    model_endpoint: Mapped[str] = mapped_column(String)
    created: Mapped[datetime] = mapped_column(TIMESTAMP)
    updated: Mapped[datetime] = mapped_column(TIMESTAMP)
    status: Mapped[str] = mapped_column(String)

    threads: Mapped[List["ThreadModel"]] = relationship(
        "ThreadModel", back_populates="session", cascade="all, delete-orphan"
    )
    settings: Mapped["SessionSettingsModel"] = relationship(
        "SessionSettingsModel", back_populates="session", uselist=False
    )


class SessionSettingsModel(Base):
    """
    Represents a session configuration table in the database.

    Attributes:
        id (Mapped[str]): The primary key, the UUID.
        playground_session_id (Mapped[str]): Foreign key linking to the SessionModel table.
        model_params: (Mapped[Dict]): The model related parameters stored as JSON.
            - max_tokens: int
            - temperature: float
            - top_k: int
            - top_p: float
    """

    __tablename__ = "playground_session_settings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    playground_session_id: Mapped[str] = mapped_column(
        String, ForeignKey("playground_session.id"), unique=True
    )
    model_params: Mapped[dict] = mapped_column(JSON)

    session: Mapped["SessionModel"] = relationship(
        "SessionModel", back_populates="settings", uselist=False
    )


class ThreadModel(Base):
    """
    Represents a thread table in the database.


    Attributes:
        id (Mapped[str]): The primary key, the UUID.
        playground_session_id (Mapped[str]): Foreign key linking to the SessionModel table.
        name (Mapped[str]): The name of the thread.
        created (Mapped[datetime]): The creating timestamp of the thread.
        updated (Mapped[datetime]): The updating timestamp of the thread.
        status (Mapped[str]): The status of the thread.
    """

    __tablename__ = "playground_thread"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    playground_session_id: Mapped[str] = mapped_column(
        String, ForeignKey("playground_session.id")
    )
    name: Mapped[str] = mapped_column(String)
    created: Mapped[datetime] = mapped_column(TIMESTAMP)
    updated: Mapped[datetime] = mapped_column(TIMESTAMP)
    status: Mapped[str] = mapped_column(String)

    session: Mapped["SessionModel"] = relationship(
        "SessionModel", back_populates="threads"
    )
    messages: Mapped[List["MessageModel"]] = relationship(
        "MessageModel", back_populates="thread", cascade="all, delete-orphan"
    )


class MessageModel(Base):
    """
    Represents a message table in the database.

    Attributes:
        id (Mapped[str]): The primary key, the UUID.
        parent_id (Mapped[str]): The parent message.
        playground_thread_id (Mapped [str]): Foreign key linking to the ThreadModel table.
        content (Mapped[str]): The message text.
        payload (Mapped[dict]): The payload info.
        model_params (Mapped[dict]): The model parameters.
        created (Mapped[datetime]): The timestamp of the request.
        updated (Mapped[datetime]): The timestamp of the request.
        status (Mapped[str]): The status of the request.
        rate (Mapped[int]): The rate of the response. [-1, 0, 1].
        role (Mapped[str]): The role of the message. [system, user]
    """

    __tablename__ = "playground_message"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    parent_id: Mapped[str] = mapped_column(String, nullable=True)
    playground_thread_id: Mapped[str] = mapped_column(
        String, ForeignKey("playground_thread.id")
    )
    content: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)
    model_params: Mapped[dict] = mapped_column(JSON)
    created: Mapped[datetime] = mapped_column(TIMESTAMP)
    updated: Mapped[datetime] = mapped_column(TIMESTAMP)
    status: Mapped[str] = mapped_column(String)
    rate: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String)

    thread: Mapped["ThreadModel"] = relationship(
        "ThreadModel", back_populates="messages"
    )
