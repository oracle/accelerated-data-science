# #!/usr/bin/env python
# # -*- coding: utf-8 -*--

# # Copyright (c) 2024 Oracle and/or its affiliates.
# # Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# from dataclasses import dataclass

# from tornado.web import HTTPError

# from ads.aqua.extension.base_handler import AquaAPIhandler
# from ads.aqua.playground.entities import Message
# from ads.aqua.playground.playground import SessionApp, ThreadApp
# from ads.common.serializer import DataClassSerializable
# from ads.common.utils import batch_convert_case


# class Errors(str):
#     INVALID_INPUT_DATA_FORMAT = "Invalid format of input data."
#     NO_INPUT_DATA = "No input data provided."
#     MISSING_REQUIRED_PARAMETER = "Missing required parameter: '{}'"


# @dataclass
# class NewSessionRequest(DataClassSerializable):
#     """Dataclass representing the request on creating a new session."""

#     model_id: str = None


# class AquaPlaygroundSessionHandler(AquaAPIhandler):
#     """
#     Handles the management and interaction with Playground sessions.

#     Methods
#     -------
#     get(self, id="")
#         Retrieves a list of sessions or a specific session by ID.
#     post(self, *args, **kwargs)
#         Creates a new playground session.
#     read(self, id: str)
#         Reads the detailed information of a specific Playground session.
#     list(self)
#         Lists all the playground sessions.

#     Raises
#     ------
#     HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
#     """

#     def get(self, id=""):
#         """
#         Retrieve a list of all sessions or a specific session by its ID.

#         Parameters
#         ----------
#         id: (str, optional)
#             The ID of the session to retrieve. Defaults to an empty string,
#             which implies fetching all sessions.

#         Returns
#         -------
#         The session data or a list of sessions.
#         """
#         if not id:
#             return self.list()
#         return self.read(id)

#     def read(self, id: str):
#         """Read the information of a Playground session."""
#         try:
#             return self.finish(SessionApp().get(id=id, include_threads=True))
#         except Exception as ex:
#             raise HTTPError(500, str(ex))

#     def list(self):
#         """List playground sessions."""
#         try:
#             return self.finish(SessionApp().list())
#         except Exception as ex:
#             raise HTTPError(500, str(ex))

#     def post(self, *args, **kwargs):
#         """
#         Creates a new Playground session by model ID.
#         The session data is extracted from the JSON body of the request.
#         If session for given model ID exists, then the existing session will be returned.

#         Raises
#         ------
#         HTTPError
#             If the input data is invalid or missing, or if an error occurs during session creation.
#         """
#         try:
#             input_data = self.get_json_body()
#         except Exception as ex:
#             raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT)

#         if not input_data:
#             raise HTTPError(400, Errors.NO_INPUT_DATA)

#         new_session_request = NewSessionRequest.from_dict(
#             batch_convert_case(input_data, to_fmt="snake")
#         )

#         if not new_session_request.model_id:
#             raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("modelId"))

#         try:
#             self.finish(SessionApp().create(model_id=new_session_request.model_id))
#         except Exception as ex:
#             raise HTTPError(500, str(ex))


# class AquaPlaygroundThreadHandler(AquaAPIhandler):
#     """
#     Handles the management and interaction with Playground threads.

#     Methods
#     -------
#     get(self, thread_id="")
#         Retrieves a list of threads or a specific thread by ID.
#     post(self, *args, **kwargs)
#         Creates a new playground thread.
#     delete(self)
#         Deletes (soft delete) a specified thread by ID.
#     read(self, thread_id: str)
#         Reads the detailed information of a specific Playground thread.
#     list(self)
#         Lists all the threads in a session.

#     Raises
#     ------
#     HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
#     """

#     def get(self, thread_id: str = ""):
#         """
#         Retrieve a list of all threads or a specific thread by its ID.

#         Parameters
#         ----------
#         thread_id (str, optional)
#             The ID of the thread to retrieve. Defaults to an empty string,
#             which implies fetching all threads.

#         Returns
#         -------
#         The thread data or a list of threads.
#         """
#         if not thread_id:
#             return self.list()
#         return self.read(thread_id)

#     def read(self, thread_id: str):
#         """Read the information of a playground thread."""
#         try:
#             return self.finish(
#                 ThreadApp().get(thread_id=thread_id, include_messages=True)
#             )
#         except Exception as ex:
#             raise HTTPError(500, str(ex))

#     def list(self):
#         """
#         List playground threads.

#         Args
#         ----
#         session_id: str
#             The ID of the session to list associated threads.
#         """
#         session_id = self.get_argument("session_id")
#         try:
#             return self.finish(ThreadApp().list(session_id=session_id))
#         except Exception as ex:
#             raise HTTPError(500, str(ex))

#     def post(self, *args, **kwargs):
#         """
#         Creates a new Playground thread.
#         The thread data is extracted from the JSON body of the request.

#         Raises
#         ------
#         HTTPError
#             If the input data is invalid or missing, or if an error occurs during thread creation.
#         """
#         try:
#             input_data = self.get_json_body()
#         except Exception as ex:
#             raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT)

#         if not input_data:
#             raise HTTPError(400, Errors.NO_INPUT_DATA)

#         try:
#             message_obj = Message.from_dict(
#                 batch_convert_case(input_data, to_fmt="snake")
#             )

#             system_message = ThreadApp().post_message(
#                 message=message_obj.content,
#                 thread_id=message_obj.thread_id,
#                 session_id=message_obj.session_id,
#                 model_params=message_obj.model_params,
#             )

#             self.finish(
#                 ThreadApp().get(
#                     thread_id=system_message.thread_id, include_messages=True
#                 )
#             )
#         except Exception as ex:
#             raise HTTPError(500, str(ex))

#     def delete(self):
#         """
#         Deletes (soft delete) the thread by ID.

#         Args
#         ----
#         thread_id: str
#             The ID of the thread to be deleted.
#         """
#         thread_id = self.get_argument("threadId")
#         if not thread_id:
#             raise HTTPError(
#                 400, Errors.Errors.MISSING_REQUIRED_PARAMETER.format("threadId")
#             )

#         # Only soft deleting with updating a status field.
#         try:
#             ThreadApp().deactivate(thread_id=thread_id)
#             self.set_status(204)  # no content
#             self.finish()
#         except Exception as ex:
#             raise HTTPError(500, str(ex))


# __handlers__ = [
#     ("playground/session/?([^/]*)", AquaPlaygroundSessionHandler),
#     ("playground/thread/?([^/]*)", AquaPlaygroundThreadHandler),
# ]
