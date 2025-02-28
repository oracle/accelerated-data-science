#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import threading
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import Callable, Dict, List, Union

from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketHandler

from ads.aqua.common.task_status import TaskStatus
from ads.common.extended_enum import ExtendedEnum

logger = getLogger(__name__)


class TaskNameEnum(ExtendedEnum):
    REGISTRATION_STATUS = "REGISTRATION_STATUS"


@dataclass
class Task:
    task_name: TaskNameEnum = None
    task_id: str = None


@dataclass
class Status(Task):
    task_status: TaskStatus = None


@dataclass
class Subscriber(Task):
    subscriber: WebSocketHandler = None


@dataclass
class RegistrationStatus(Status):
    task_name: TaskNameEnum = TaskNameEnum.REGISTRATION_STATUS


@dataclass
class RegistrationSubscriber(Subscriber):
    task_name: TaskNameEnum = TaskNameEnum.REGISTRATION_STATUS


@dataclass
class StatusSubscription:
    task_status_list: List[TaskStatus] = field(default_factory=list)
    subscribers: List[Subscriber] = field(default_factory=list)


class StatusTracker:
    lock = threading.RLock()
    """
    Maintains a mapping of task statuses and subscribers for notifications.
    Example:
        {
            "REGISTRATION_STATUS": {
                "sample-task-id": StatusSubscription(
                    task_status_list=[TaskStatus(state="MODEL_DOWNLOAD_INPROGRESS", message="1 out of 10 files downloaded")],
                    subscribers=[Subscriber(subscriber=websocket123)]
                )
            }
        }
    """
    status: Dict[TaskNameEnum, Dict[str, StatusSubscription]] = {}

    @staticmethod
    def get_latest_status(
        task_name: TaskNameEnum, task_id: str
    ) -> Union[TaskStatus, None]:
        """Returns latest task status if availble, else returns None"""
        task_list = []
        logger.info(f"Status dump: {StatusTracker.status}")
        with StatusTracker.lock:
            task_list = (
                StatusTracker.status.get(task_name, {})
                .get(task_id, StatusSubscription())
                .task_status_list
            )
        return task_list[-1] if task_list else None

    @staticmethod
    def get_statuses(task_name: TaskNameEnum, task_id: str) -> Union[TaskStatus, None]:
        """Returns latest task status if availble, else returns None"""
        with StatusTracker.lock:
            return (
                StatusTracker.status.get(task_name, {})
                .get(task_id, StatusSubscription())
                .task_status_list
            )

    @staticmethod
    def add_status(status: Status, notify=True):
        """Appends to the status list. Notifies the status to all the subcribers"""
        logger.info(f"status: {status}")
        with StatusTracker.lock:
            if status.task_name not in StatusTracker.status:
                StatusTracker.status[status.task_name] = {
                    status.task_id: StatusSubscription(
                        task_status_list=[status.task_status]
                    )
                }
            elif status.task_id in StatusTracker.status[status.task_name]:
                StatusTracker.status[status.task_name][
                    status.task_id
                ].task_status_list.append(status.task_status)
            else:
                StatusTracker.status[status.task_name][status.task_id] = (
                    StatusSubscription(task_status_list=[status.task_status])
                )
        # Since there is a task id, Notify subscribers if any
        if notify:
            StatusTracker.notify_latest_to_all(
                task_name=status.task_name, task_id=status.task_id
            )

    @staticmethod
    def notify_latest_to_all(task_name: TaskNameEnum, task_id: str):
        """Notify the latest task status to all the subscribers"""
        task_status = StatusTracker.get_latest_status(
            task_name=task_name, task_id=task_id
        )
        logger.info(f"status: {task_status}")
        subscribers = []
        with StatusTracker.lock:
            subscribers = (
                StatusTracker.status.get(task_name, {})
                .get(task_id, StatusSubscription())
                .subscribers
            )
        for subscriber in subscribers:
            StatusTracker.send_message(status=task_status, subscriber=subscriber)

    @staticmethod
    def notify(task_name: TaskNameEnum, subscriber: Subscriber, latest_only=True):
        """Notify the subscriber of all the status"""
        if latest_only:
            task_status = StatusTracker.get_latest_status(
                task_name=task_name, task_id=subscriber.task_id
            )
        else:
            task_status = StatusTracker.get_statuses(
                task_name=task_name, task_id=subscriber.task_id
            )
        logger.info(task_status)
        StatusTracker.send_message(status=task_status, subscriber=subscriber)

    @staticmethod
    def send_message(status: TaskStatus, subscriber: Subscriber):
        if (
            subscriber
            and subscriber.ws_connection
            and subscriber.ws_connection.stream.socket
        ):
            try:
                subscriber.write_message(status.to_json())
            except Exception as e:
                print(e)
                IOLoop.current().add_callback(
                    lambda: subscriber.write_message(status.to_json())
                )

    @staticmethod
    def add_subscriber(subscriber: Subscriber, notify_latest_status=True):
        """Appends to the subscriber list"""
        with StatusTracker.lock:
            if subscriber.task_name not in StatusTracker.status:
                StatusTracker.status[subscriber.task_name] = {
                    subscriber.task_id: StatusSubscription(
                        subscribers=[subscriber.subscriber]
                    )
                }
            elif subscriber.task_id in StatusTracker.status[subscriber.task_name]:
                StatusTracker.status[subscriber.task_name][
                    subscriber.task_id
                ].subscribers.append(subscriber.subscriber)
            else:
                StatusTracker.status[subscriber.task_name][subscriber.task_id] = (
                    StatusSubscription(subscribers=[subscriber.subscriber])
                )
        if notify_latest_status:
            StatusTracker.notify(
                task_name=subscriber.task_name, task_id=subscriber.task_id
            )

    @staticmethod
    def prepare_status_callback(
        task_name: TaskNameEnum, task_id: str
    ) -> Callable[[TaskStatus], None]:
        def callback(task_name: TaskNameEnum, task_id: str, status: TaskStatus):
            st = Status(task_name=task_name, task_id=task_id, task_status=status)
            StatusTracker.add_status(st)

        return partial(callback, task_name=task_name, task_id=task_id)
