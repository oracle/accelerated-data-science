#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains Mixins for integrating OCI data models
"""
import inspect
import json
import logging
import os
import re
import time
import traceback
from datetime import date, datetime
from typing import Callable, Optional, Union
from enum import Enum

import oci
import yaml
from ads.common import auth
from ads.common.decorator.utils import class_or_instance_method
from ads.common.utils import camel_to_snake, get_progress_bar
from ads.config import COMPARTMENT_OCID
from dateutil import tz
from dateutil.parser import parse
from oci._vendor import six

logger = logging.getLogger(__name__)

LIFECYCLE_STOP_STATE = ("SUCCEEDED", "FAILED", "CANCELED", "DELETED")
WORK_REQUEST_STOP_STATE = ("SUCCEEDED", "FAILED", "CANCELED")
DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
DEFAULT_WORKFLOW_STEPS = 2


class MergeStrategy(Enum):
    OVERRIDE = "override"
    MERGE = "merge"


class OCIModelNotExists(Exception):   # pragma: no cover
    pass


class OCIClientMixin:
    """Mixin class for representing OCI resource/service with OCI client.

    Most OCI requests requires a client for making the connection.
    Usually the same client will be used for the requests related the same resource/service type.
    This Mixin adds a "client" property to simplify accessing the client.
    The actual client will be initialize lazily so that it is not required for a sub-class
    To use the client, sub-class should override the init_client() method pass in the "client" keyword argument.
    For example:

    @class_or_instance_method
    def init_client(cls, **kwargs) -> oci.logging.LoggingManagementClient:
        return cls._init_client(client=oci.logging.LoggingManagementClient, **kwargs)

    Instance methods in the sub-class can use self.client to access the client.
    The init_client() method is a class method used to create the client.
    Any class method using the client should use init_client() to create the client.
    The call to this method may come from an instance or a class.
    When the method is called from a class,
        the default authentication configured at ADS level with ads.set_auth() will be used.
    When the method is called from an instance,
        the config, signer and kwargs specified when initializing the instance will be used.
    The sub-class's __init__ method should take config, signer and client_kwargs as argument,
        then pass them to the __init__ method of this class.
    This allows users to override the authentication and client initialization parameters.
    For example, log_group = OCILogGroup(config=config, signer=signer, client_kwargs=kwargs)
    """

    config = None
    signer = None
    kwargs = None

    @class_or_instance_method
    def _get_auth(cls):
        client_kwargs = dict(retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        if cls.kwargs:
            client_kwargs.update(cls.kwargs)

        if cls.config is None and cls.signer is None:
            oci_auth = auth.default_signer(client_kwargs)
        elif not cls.signer and cls.config:
            oci_auth = {"config": cls.config, "client_kwargs": client_kwargs}
        else:
            oci_auth = {
                "config": cls.config,
                "signer": cls.signer,
                "client_kwargs": client_kwargs,
            }
        return oci_auth

    @class_or_instance_method
    def init_client(cls, **kwargs):
        """Initializes the OCI client specified in the "client" keyword argument
        Sub-class should override this method and call cls._init_client(client=OCI_CLIENT)

        Parameters
        ----------
        **kwargs :
            Additional keyword arguments for initializing the OCI client.

        Returns
        -------
            An instance of OCI client.

        """
        return cls._init_client(**kwargs)

    @class_or_instance_method
    def _init_client(cls, client, **kwargs):
        """Initializes the OCI client specified in the "client" argument

        Parameters
        ----------
        client :
            The OCI client class to be initialized, e.g., oci.data_science.DataScienceClient
        **kwargs :
            Additional keyword arguments for initializing the OCI client.

        Returns
        -------
            An instance of OCI client.

        """
        auth = cls._get_auth()
        auth_kwargs = auth.pop("client_kwargs", {})
        if kwargs:
            auth_kwargs.update(kwargs)
        auth.update(auth_kwargs)
        return client(**auth)

    @class_or_instance_method
    def create_instance(obj, *args, **kwargs):
        """Creates an instance using the same authentication as the class or an existing instance.
        If this method is called by a class, the default ADS authentication method will be used.
        If this method is called by an instance, the authentication method set in the instance will be used.
        """
        # Here cls could be a class or an instance.
        # If it is a class, it can be used to create the new instance directly.
        # If it is an instance, the new instance should be created using cls.__class__.
        # Calling the classmethod _create_instance() will make sure that class will be used.
        return obj._create_instance(
            config=obj.config,
            signer=obj.signer,
            client_kwargs=obj.kwargs,
            *args,
            **kwargs,
        )

    @classmethod
    def _create_instance(cls, *args, **kwargs):
        """Initialize an instance."""
        return cls(*args, **kwargs)

    def __init__(self, config=None, signer=None, client_kwargs=None) -> None:
        """Initializes a service/resource with OCI client as a property.
        If config or signer is specified, it will be used to initialize the OCI client.
        If neither of them is specified, the client will be initialized with ads.common.auth.default_signer.
        If both of them are specified, both of them will be passed into the OCI client,
            and the authentication will be determined by OCI Python SDK.

        Parameters
        ----------
        config : dict, optional
            OCI API key config dictionary, by default None.
        signer : oci.signer.Signer, optional
            OCI authentication signer, by default None.
        client_kwargs : dict, optional
            Additional keyword arguments for initializing the OCI client.
        """
        super().__init__()
        self.config = config
        self.signer = signer
        self.kwargs = client_kwargs
        self._client = None

    @property
    def auth(self) -> dict:
        """The ADS authentication config used to initialize the client.
        This auth has the same format as those obtained by calling functions in ads.common.auth.
        The config is a dict containing the following key-value pairs:
        config: The config contains the config loaded from the configuration loaded from `oci_config`.
        signer: The signer contains the signer object created from the api keys.
        client_kwargs: client_kwargs contains the `client_kwargs` that was passed in as input parameter.
        """
        return self._get_auth()

    @property
    def client(self):
        """OCI client"""
        if self._client is None:
            self._client = self.init_client()
        return self._client


class OCISerializableMixin(OCIClientMixin):
    """Mixin class containing OCI serialization/de-serialization methods.
    These methods are copied and modified from the OCI BaseClient.

    """

    type_mappings = None

    def serialize(self):
        """Serialize the model to a dictionary that is ready to be send to OCI API.

        Returns
        -------
        dict
            A dictionary that is ready to be send to OCI API.

        """
        return self.client.base_client.sanitize_for_serialization(self)

    @staticmethod
    def _parse_kwargs(attribute_map: dict, **kwargs):
        """Parse kwargs to make all key in camel format."""
        parsed_kwargs = {}
        for key, val in kwargs.items():
            if key in attribute_map:
                parsed_kwargs[attribute_map[key]] = val
            else:
                parsed_kwargs[key] = val

        return parsed_kwargs

    @class_or_instance_method
    def deserialize(cls, data, to_cls):
        """De-serialize data from dictionary to an OCI model"""
        if cls.type_mappings is None:
            cls.type_mappings = cls.init_client().base_client.type_mappings

        if data is None:
            return None

        # This is a work around for enums not being present
        # in the type mappings.
        # See OraclePythonSdkCodegen removeEnumsFromModelGeneration().
        if to_cls in cls.type_mappings:
            to_cls = cls.type_mappings[to_cls]

            if isinstance(data, dict):
                data = cls._parse_kwargs(to_cls().attribute_map, **data)
        else:
            return cls.__deserialize_primitive(data, to_cls)

        if hasattr(to_cls, "get_subtype"):
            # Return the object as is if it is already deserialized.
            if isinstance(data, to_cls) or issubclass(data.__class__, to_cls):
                return data

            # Use the discriminator value to get the correct subtype.
            to_cls = to_cls.get_subtype(data)  # get_subtype returns a str
            to_cls = cls.type_mappings[to_cls]

            # kwargs needs to be parsed again as there are more attributes in the sub types.
            if isinstance(data, dict):
                data = cls._parse_kwargs(to_cls().attribute_map, **data)

        if to_cls in [int, float, six.u, bool]:
            return cls.__deserialize_primitive(data, to_cls)
        elif to_cls == object:
            return data
        elif to_cls == date:
            return cls.__deserialize_date(data)
        elif to_cls == datetime:
            return cls.__deserialize_datetime(data)
        else:
            return cls.__deserialize_model(data, to_cls)

    @classmethod
    def __deserialize_model(cls, data, to_cls):
        """De-serializes list or dict to model."""
        if isinstance(data, to_cls):
            return data

        instance = to_cls()

        for attr, attr_type in instance.swagger_types.items():
            property = instance.attribute_map[attr]
            if property in data:
                value = data[property]
                setattr(instance, attr, cls.deserialize(value, attr_type))

        return instance

    @staticmethod
    def __deserialize_primitive(data, cls):
        """De-serializes string to primitive type."""
        try:
            value = cls(data)
        except UnicodeEncodeError:
            value = data
        except TypeError:
            value = data
        return value

    @staticmethod
    def __deserialize_date(string):
        """De-serializes string to date."""
        try:
            return parse(string).date()
        except ImportError:
            return string
        except ValueError:
            raise Exception("Failed to parse `{0}` into a date object".format(string))

    @staticmethod
    def __deserialize_datetime(string):
        """De-serializes string to datetime.

        The string should be in iso8601 datetime format.
        """
        if isinstance(string, datetime):
            return string
        try:
            # If this parser creates a date without raising an exception
            # then the time zone is utc and needs to be set.
            naivedatetime = datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%fZ")
            awaredatetime = naivedatetime.replace(tzinfo=tz.tzutc())
            return awaredatetime

        except ValueError:
            try:
                return parse(string)
            except ImportError:
                return string
            except ValueError:
                raise Exception(
                    "Failed to parse `{0}` into a datetime object".format(string)
                )
        except ImportError:
            return string


class OCIModelMixin(OCISerializableMixin):
    """Mixin class to operate OCI model.
    OCI resources are represented by models in the OCI Python SDK.

    Unifying OCI models for the same resource
    -----------------------------------------
    OCI SDK uses different models to represent the same resource for different operations.
    For example, CreateLogDetails is used when creating a log resource,
        while LogSummary is returned when listing the log resources.
    However, both CreateLogDetails and LogSummary have the same commonly used attribute like
        compartment_id, display_name, log_type, etc.
    In general, there is a class with a super set of all properties.
    For example, the Log class contains all properties of CreateLogDetails and LogSummary,
        as well as other classes representing an OCI log resource.
    A subclass can be implemented with this Mixin to unify the OCI models,
        so that all properties are available to the user.
    For example, if we define the Mixin model as ``class OCILog(OCIModelMixin, oci.logging.models.Log)``,
        users will be able to access properties like ``OCILog().display_name``
    Since this sub-class contains all the properties, it can be converted to any related OCI model in an operation.
    For example, we can create ``CreateLogDetails`` from ``OCILog`` by extracting a subset of the properties.
    When listing the resources, properties from ``LogSummary`` can be used to update
        the corresponding properties of ``OCILog``.
    Such convertion can be done be the generic methods provided by this Mixin.
    Although OCI SDK accepts dictionary (JSON) data instead of objects like CreateLogDetails when creating or
        updating the resource, the structure and keys of the dictionary is not easy for a user to remember.
    It is also unnecessary for the users to construct the entire dictionary if they only want to update a single value.

    This Mixin class should be the first parent as the class from OCI SDK does not call ``super().__init__()``
        in its ``__init__()`` constructor.
    Mixin properties may not be intialized correctly if ``super().__init__()`` is not called.


    Provide generic methods for CRUDL operations
    --------------------------------------------
    Since OCI SDK uses different models in CRUDL operations,
        this Mixin provides the following method to convert between them.
    An OCI model instance can be any OCI model of the resource containing some properties, e.g. LogSummary
    ``from_oci_model()`` static method can be used to initialize a new instance from an OCI model instance.
    ``update_from_oci_model()`` can be used to update the existing properties from an OCI model instance.
    ``to_oci_model()`` can be used to extract properties from the Mixin model to OCI model.

    """

    # Regex pattern matching the module name of an OCI model.
    OCI_MODEL_PATTERN = r"oci.[^.]+\.models[\..*]?"
    # Constants
    CONS_COMPARTMENT_ID = "compartment_id"

    @staticmethod
    def check_compartment_id(compartment_id: Optional[str]) -> str:
        """Checks if a compartment ID has value and
            return the value from NB_SESSION_COMPARTMENT_OCID environment variable if it is not specified.

        Parameters
        ----------
        compartment_id : str
            Compartment OCID or None

        Returns
        -------
        type
            str: Compartment OCID

        Raises
        ------
        ValueError
            compartment_id is not specified and NB_SESSION_COMPARTMENT_OCID environment variable is not set

        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        if not compartment_id:
            raise ValueError("Specify compartment OCID.")
        return compartment_id

    @class_or_instance_method
    def list_resource(
        cls, compartment_id: str = None, limit: int = 0, **kwargs
    ) -> list:
        """Generic method to list OCI resources

        Parameters
        ----------
        compartment_id : str
            Compartment ID of the OCI resources. Defaults to None.
            If compartment_id is not specified,
            the value of NB_SESSION_COMPARTMENT_OCID in environment variable will be used.
        limit : int
            The maximum number of items to return. Defaults to 0, All items will be returned
        **kwargs :
            Additional keyword arguments to filter the resource.
            The kwargs are passed into OCI API.

        Returns
        -------
        list
            A list of OCI resources

        Raises
        ------
        NotImplementedError
            List method is not supported or implemented.

        """
        if limit:
            items = cls._find_oci_method("list")(
                cls.check_compartment_id(compartment_id), limit=limit, **kwargs
            ).data
        else:
            items = oci.pagination.list_call_get_all_results(
                cls._find_oci_method("list"),
                cls.check_compartment_id(compartment_id),
                **kwargs,
            ).data
        return [cls.from_oci_model(item) for item in items]

    @classmethod
    def _find_oci_parent(cls):
        """Finds the parent OCI model class

        Parameters
        ----------

        Returns
        -------
        class
            An OCI model class

        Raises
        ------
        AttributeError
            If the class if not inherit from an OCI model.

        """
        oci_model = None
        for parent in inspect.getmro(cls):
            if re.match(OCIModelMixin.OCI_MODEL_PATTERN, parent.__module__):
                oci_model = parent
                break
        if not oci_model:
            raise AttributeError(f"{cls.__name__} is not inherited from an OCI model.")
        return oci_model

    @class_or_instance_method
    def _find_oci_method(cls, method: str) -> Callable:
        """Finds the OCI method for operations like get/list

        Parameters
        ----------
        method : str
            The operation to be performed, e.g. get, list, delete


        Returns
        -------
        callable
            The method from OCI client to perform the operation.

        """
        client = cls.init_client()

        user_define_method = f"oci_{method}_method".upper()
        if hasattr(cls, user_define_method) and getattr(cls, user_define_method):
            return getattr(cls, user_define_method).__get__(client, client.__class__)

        oci_class = cls._find_oci_parent()
        if method == "list":
            method_name = f"{method}_{camel_to_snake(oci_class.__name__)}s"
        else:
            method_name = f"{method}_{camel_to_snake(oci_class.__name__)}"

        if hasattr(client, method_name):
            return getattr(client, method_name)

        raise NotImplementedError(
            f"{str(oci_class)} does not have {method_name} method. "
            f"Define {user_define_method} in {str(cls)}"
        )

    @class_or_instance_method
    def from_oci_model(cls, oci_instance):
        """Initialize an instance from an instance of OCI model.

        Parameters
        ----------
        oci_instance :
            An instance of an OCI model.

        """
        oci_model = cls._find_oci_parent()
        kwargs = {}
        for attr in oci_model().swagger_types.keys():
            if hasattr(oci_instance, attr):
                kwargs[attr] = getattr(oci_instance, attr)
        instance = cls.create_instance(**kwargs)
        if hasattr(oci_instance, "attribute_map"):
            instance._oci_attributes = oci_instance.attribute_map
        return instance

    @class_or_instance_method
    def from_dict(cls, data):
        """Initialize an instance from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the properties to initialize the class.

        """
        return cls.create_instance(**data)

    @class_or_instance_method
    def deserialize(cls, data: dict, to_cls: str = None):
        """Deserialize data

        Parameters
        ----------
        data : dict
            A dictionary containing the data to be deserialized.

        to_cls : str
            The name of the OCI model class to be initialized using the data.
            The OCI model class must be from the same OCI service of the OCI client (self.client).
            Defaults to None, the parent OCI model class name will be used
            if current class is inherited from an OCI model.
            If parent OCI model class is not found or not from the same OCI service,
            the data will be returned as is.

        """
        if to_cls is None:
            oci_model = cls._find_oci_parent()
            to_cls = oci_model.__name__
        return super().deserialize(data, to_cls)

    @class_or_instance_method
    def _get(cls, ocid):
        get_method = cls._find_oci_method("get")
        return get_method(ocid).data

    @class_or_instance_method
    def from_ocid(cls, ocid: str):
        """Initializes an object from OCID

        Parameters
        ----------
        ocid : str
            The OCID of the object

        """
        oci_instance = cls._get(ocid)
        return cls.from_oci_model(oci_instance)

    def __init__(
        self,
        config: dict = None,
        signer: oci.signer.Signer = None,
        client_kwargs: dict = None,
        **kwargs,
    ) -> None:
        # Initialize an empty object
        super().__init__(config=config, signer=signer, client_kwargs=client_kwargs)

        if kwargs:
            self.update_from_oci_model(
                self.deserialize(kwargs), merge_strategy=MergeStrategy.MERGE
            )
        # When from_oci_model is called to initialize the object from an OCI model,
        # _oci_attributes stores the attribute names from the OCI model
        # This is used to determine if an additional get request should be called to get additional attributes.
        # For example, some attributes are not included in the results from OCI list request.
        # When user access such attributes, an additional get request should be made to get the data from OCI.
        # See __getattribute__() for the implementation.
        self._oci_attributes = {}

    @property
    def name(self) -> str:
        """Gets the name of the object."""
        if hasattr(self, "display_name"):
            return getattr(self, "display_name")
        return ""

    @name.setter
    def name(self, value: str):
        """Sets the name of the object.

        Parameters
        ----------
        value : str
            The name of the object.
        """
        setattr(self, "display_name", value)

    def load_properties_from_env(self):
        """Loads properties from the environment"""
        env_var_mapping = {
            "project_id": ["PROJECT_OCID"],
            OCIModelMixin.CONS_COMPARTMENT_ID: [
                "JOB_RUN_COMPARTMENT_OCID",
                "NB_SESSION_COMPARTMENT_OCID",
            ],
        }
        for attr, env_names in env_var_mapping.items():
            try:
                env_value = next(
                    os.environ.get(env_name)
                    for env_name in env_names
                    if os.environ.get(env_name, None) is not None
                )
                if getattr(self, attr, "") is None:
                    setattr(self, attr, env_value)
            except:
                pass

    def to_oci_model(self, oci_model):
        """Converts the object into an instance of OCI data model.

        Parameters
        ----------
        oci_model : class or str
            The OCI model to be converted to. This can be a string of the model name.
        type_mapping : dict
            A dictionary mapping the models.
            Returns: An instance of the oci_model

        Returns
        -------

        """
        if isinstance(oci_model, str):
            oci_model_name = oci_model
        else:
            oci_model_name = oci_model.__name__
        data = json.dumps(self.to_dict()).encode("utf8")
        return self.client.base_client.deserialize_response_data(data, oci_model_name)

    @staticmethod
    def flatten(data: dict) -> dict:
        """Flattens a nested dictionary.

        Parameters
        ----------
        data : A nested dictionary


        Returns
        -------
        dict
            The flattened dictionary.

        """
        flatten_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flatten_dict.update(OCIModelMixin.flatten(value))
            else:
                flatten_dict[key] = value
        return flatten_dict

    def to_dict(self, flatten: bool = False) -> dict:
        """Converts the properties to a dictionary

        Parameters
        ----------
        flatten :
             (Default value = False)

        Returns
        -------

        """
        data = self.serialize()
        if flatten:
            data = OCIModelMixin.flatten(data)
        return data

    def update_from_oci_model(
        self, oci_model_instance, merge_strategy: MergeStrategy = MergeStrategy.OVERRIDE
    ):
        """Updates the properties from OCI model with the same properties.

        Parameters
        ----------
        oci_model_instance :
            An instance of OCI model, which should have the same properties of this class.
        """
        for attr in self.swagger_types.keys():
            if (
                hasattr(oci_model_instance, attr)
                and getattr(oci_model_instance, attr)
                and (
                    not hasattr(self, attr)
                    or not getattr(self, attr)
                    or merge_strategy == MergeStrategy.OVERRIDE
                )
            ):
                setattr(self, attr, getattr(oci_model_instance, attr))

        if hasattr(oci_model_instance, "attribute_map"):
            self._oci_attributes = oci_model_instance.attribute_map
        return self

    def sync(self, merge_strategy: MergeStrategy = MergeStrategy.OVERRIDE):
        """Refreshes the properties of the object from OCI"""
        return self.update_from_oci_model(
            self.from_ocid(self.id), merge_strategy=merge_strategy
        )

    def delete(self):
        """Deletes the resource"""
        delete_method = self._find_oci_method("delete")
        delete_method(self.id)
        return self

    def __getattribute__(self, name: str):
        """Returns an attribute value of the resource.

        This method will try to sync the values from OCI service when it is not already available locally.
        Some attribute value may not be available locally if the previous OCI API call returns an OCI model
        that contains only a subset of the attributes.
        For example, JobSummary model contains only a subset of the attributes from the Job model.

        Parameters
        ----------
        name : str
            Attribute name.
        """
        skip_lookup = ["id", "attribute_map", "_oci_attributes"]

        if name in skip_lookup or name.startswith("_"):
            return super().__getattribute__(name)

        # Ignore if _oci_attributes is not initialized
        if not hasattr(self, "_oci_attributes"):
            return super().__getattribute__(name)

        if (
            hasattr(self, "attribute_map")
            and name in self.attribute_map
            and name not in self._oci_attributes
            and hasattr(self, "id")
            and self.id
        ):
            # Do not sync if it is in the sync process
            stack = traceback.extract_stack()
            for frame in reversed(stack):
                if frame.name == "sync":
                    return super().__getattribute__(name)
            # Sync only if there is no value
            if not super().__getattribute__(name):
                try:
                    self.sync(merge_strategy=MergeStrategy.MERGE)
                except oci.exceptions.ServiceError as ex:
                    # 400 errors are usually cause by the user
                    if ex.status == 400:
                        logger.error("%s - %s: %s", self.__class__, ex.code, ex.message)
                        self._oci_attributes = self.attribute_map
                    else:
                        logger.error(
                            "Failed to synchronize the properties of %s due to service error:\n%s",
                            self.__class__,
                            str(ex),
                        )
                except Exception as ex:
                    logger.error(
                        "Failed to synchronize the properties of %s: %s\n%s",
                        self.__class__,
                        type(ex),
                        str(ex),
                    )
        return super().__getattribute__(name)

    @property
    def status(self) -> Optional[str]:
        """Status of the object.

        Returns
        -------
        str
            Status of the object.
        """
        if not self.lifecycle_state or not self.lifecycle_state in LIFECYCLE_STOP_STATE:
            self.sync()
        return self.lifecycle_state

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

    def to_yaml(self) -> str:
        """Serializes the object into YAML string.

        Returns
        -------
        str
            YAML stored in a string.
        """
        return yaml.safe_dump(self.to_dict())


class OCIWorkRequestMixin:
    """Mixin class containing methods related to OCI work request"""

    def wait_for_work_request(
        self,
        work_request_id: str,
        wait_for_state: Union[str, tuple],
        max_wait_seconds: int = None,
        wait_interval_seconds: int = None,
    ):
        """Wait for a work request to be completed.

        Parameters
        ----------
        work_request_id : str
            OCI work request ID
        wait_for_state : str or tuple
            The state to wait for. Must be a tuple for multiple states.
        max_wait_seconds : int
            Max wait seconds for the work request. Defaults to None (Default value from OCI SDK will be used).
        wait_interval_seconds : int
            Interval in seconds between each status check. Defaults to None (Default value from OCI SDK will be used).

        Returns
        -------
        Response
            OCI API Response

        """
        result = None
        if hasattr(self.client, "get_work_request") and callable(
            getattr(self.client, "get_work_request")
        ):
            try:
                wait_period_kwargs = {}
                if max_wait_seconds is not None:
                    wait_period_kwargs["max_wait_seconds"] = max_wait_seconds
                if wait_interval_seconds is not None:
                    wait_period_kwargs["max_interval_seconds"] = wait_interval_seconds

                logger.debug(
                    "Action completed. Waiting until the work request has entered state: %s",
                    wait_for_state,
                )
                result = oci.wait_until(
                    self.client,
                    self.client.get_work_request(work_request_id),
                    "status",
                    wait_for_state,
                    **wait_period_kwargs,
                )
            except oci.exceptions.MaximumWaitTimeExceeded:
                # If we fail, we should show an error, but we should still provide the information to the customer
                logger.error(
                    "Failed to wait until the work request %s entered the specified state. Maximum wait time exceeded",
                    work_request_id,
                )
            except Exception:
                logger.error(
                    "Encountered error while waiting for work request to enter the specified state."
                )
                raise
        else:
            logger.error(
                "%s does not support wait for the work request to enter the specified state",
                str(self.client.__class__),
            )
        return result

    def get_work_request_response(
        self,
        response: str,
        wait_for_state: Union[str, tuple],
        success_state: str,
        max_wait_seconds: int = None,
        wait_interval_seconds: int = None,
        error_msg: str = "",
    ):
        if "opc-work-request-id" in response.headers:
            opc_work_request_id = response.headers.get("opc-work-request-id")
            work_request_response = self.wait_for_work_request(
                opc_work_request_id,
                wait_for_state=wait_for_state,
                max_wait_seconds=max_wait_seconds,
                wait_interval_seconds=wait_interval_seconds,
            )
            # Raise an error if the failed to create the resource.
            if work_request_response.data.status != success_state:
                raise oci.exceptions.RequestException(
                    f"{error_msg}\n" + str(work_request_response.data)
                )

        else:
            # This will likely never happen as OCI SDK will raise an error if the HTTP request is not successful.
            raise oci.exceptions.RequestException(
                f"opc-work-request-id not found in response headers: {response.headers}"
            )
        return work_request_response

    def wait_for_progress(
        self, 
        work_request_id: str, 
        num_steps: int = DEFAULT_WORKFLOW_STEPS, 
        max_wait_time: int = DEFAULT_WAIT_TIME, 
        poll_interval: int = DEFAULT_POLL_INTERVAL
    ):
        """Waits for the work request progress bar to be completed.

        Parameters
        ----------
        work_request_id: str
            Work Request OCID.
        num_steps: (int, optional). Defaults to 2.
            Number of steps for the progress indicator.
        max_wait_time: int
            Maximum amount of time to wait in seconds (Defaults to 1200).
            Negative implies infinite wait time.
        poll_interval: int
            Poll interval in seconds (Defaults to 10).

        Returns
        -------
        None
        """
        work_request_logs = []

        i = 0
        start_time = time.time()
        with get_progress_bar(num_steps) as progress:
            seconds_since = time.time() - start_time
            exceed_max_time = max_wait_time > 0 and seconds_since >= max_wait_time
            if exceed_max_time:
                logger.error(
                    f"Max wait time ({max_wait_time} seconds) exceeded."
                )
            while not exceed_max_time and (not work_request_logs or len(work_request_logs) < num_steps):
                time.sleep(poll_interval)
                new_work_request_logs = []

                try:
                    work_request = self.client.get_work_request(work_request_id).data
                    work_request_logs = self.client.list_work_request_logs(
                        work_request_id
                    ).data
                except Exception as ex:
                    logger.warn(ex)

                new_work_request_logs = (
                    work_request_logs[i:] if work_request_logs else []
                )

                for wr_item in new_work_request_logs:
                    progress.update(wr_item.message)
                    i += 1

                if work_request and work_request.status in WORK_REQUEST_STOP_STATE:
                    if work_request.status != "SUCCEEDED":
                        if new_work_request_logs:
                            raise Exception(new_work_request_logs[-1].message)
                        else:
                            raise Exception(
                                "Error occurred in attempt to perform the operation. "
                                "Check the service logs to get more details. "
                                f"{work_request}"
                            )
                    else:
                        break
            progress.update("Done")


class OCIModelWithNameMixin:
    """Mixin class to operate OCI model which contains name property."""

    @classmethod
    def from_name(cls, name: str, compartment_id: Optional[str] = None):
        """Initializes an object from name.

        Parameters
        ----------
        name: str
            The name of the object.
        compartment_id: (str, optional). Defaults to None.
            Compartment OCID of the OCI resources. If `compartment_id` is not specified,
            the value will be taken from environment variables.
        """
        res = cls.list_resource(compartment_id=compartment_id, limit=1, name=name)
        if not res:
            raise OCIModelNotExists()
        return cls.from_oci_model(res[0])
