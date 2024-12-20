#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import logging
from datetime import datetime
from typing import Any, Dict, List, Union
from uuid import UUID, uuid4

import oci
from autogen import Agent, ConversableAgent, OpenAIWrapper
from autogen.logger.base_logger import BaseLogger, LLMConfig
from autogen.logger.logger_utils import get_current_ts
from oci.monitoring import MonitoringClient
from pydantic import BaseModel, Field

import ads
import ads.config
from ads.llm.autogen.v02.loggers.utils import serialize_response

logger = logging.getLogger(__name__)


class MetricName:
    """Constants for metric name."""

    TOOL_CALL = "tool_call"
    CHAT_COMPLETION = "chat_completion_count"
    COST = "chat_completion_cost"
    SESSION_START = "session_start"
    SESSION_STOP = "session_stop"


class MetricDimension:
    """Constants for metric dimension."""

    AGENT_NAME = "agent_name"
    APP_NAME = "app_name"
    MODEL = "model"
    SESSION_ID = "session_id"
    TOOL_NAME = "tool_name"


class Metric(BaseModel):
    """Represents the metric to be logged."""

    name: str
    value: float
    timestamp: str
    dimensions: dict = Field(default_factory=dict)


class MetricLogger(BaseLogger):
    """AutoGen logger for agent metrics."""

    def __init__(
        self,
        app_name: str,
        namespace: str,
        compartment_id: str = None,
        session_id: str = None,
        region: str = None,
        resource_group: str = None,
    ):
        """Initialize the metric logger.

        Parameters
        ----------
        app_name : str
            Application name, which will be a metric dimension.
        namespace : str
            Namespace for posting the metric
        compartment_id : str, optional
            Compartment OCID for posting the metric.
            If compartment_id is not specified,
            ADS will try to fetch the compartment OCID from environment variable.
        session_id : str, optional
            Session ID to be saved as a metric dimension, by default None.
            If session_id is None, a UUID will be generated automatically.
        region : str, optional
            OCI region for posting the metric, by default None.
            If region is not specified, the region from the authentication signer will be used.
        resource_group : str, optional
            Resource group for the metric, by default None

        """
        self.app_name = app_name
        self.session_id = str(session_id or uuid4())
        self.compartment_id = compartment_id or ads.config.COMPARTMENT_OCID
        if not self.compartment_id:
            raise ValueError(
                "Unable to determine compartment OCID for metric logger."
                "Please specify the compartment_id."
            )
        self.namespace = namespace
        self.resource_group = resource_group

        # Indicate if the logger has started.
        self.started = False

        auth = ads.auth.default_signer()

        # Use the signer to determine the region if it not specified.
        signer = auth.get("signer")
        if not region:
            if hasattr(signer, "region") and signer.region:
                region = signer.region
            else:
                raise ValueError(
                    "Unable to determine the region for OCI monitoring service. "
                    "Please specify the region using the `region` argument."
                )

        self.monitoring_client = MonitoringClient(
            config=auth.get("config", {}),
            signer=signer,
            # Metrics should be submitted with the "telemetry-ingestion" endpoint instead.
            # See note here: https://docs.oracle.com/iaas/api/#/en/monitoring/20180401/MetricData/PostMetricData
            service_endpoint=f"https://telemetry-ingestion.{region}.oraclecloud.com",
        )

    def _post_metric(self, metric: Metric):
        """Posts metric to OCI monitoring."""
        # Add app_name and session_id to dimensions
        dimensions = metric.dimensions
        dimensions.update(
            {
                MetricDimension.SESSION_ID: self.session_id,
                MetricDimension.APP_NAME: self.app_name,
            }
        )
        logger.debug("Posting metrics:\n%s", str(metric))
        self.monitoring_client.post_metric_data(
            post_metric_data_details=oci.monitoring.models.PostMetricDataDetails(
                metric_data=[
                    oci.monitoring.models.MetricDataDetails(
                        namespace=self.namespace,
                        compartment_id=self.compartment_id,
                        name=metric.name,
                        dimensions=dimensions,
                        datapoints=[
                            oci.monitoring.models.Datapoint(
                                timestamp=datetime.strptime(
                                    metric.timestamp.replace(" ", "T") + "Z",
                                    "%Y-%m-%dT%H:%M:%S.%fZ",
                                ),
                                value=metric.value,
                                count=1,
                            )
                        ],
                        resource_group=self.resource_group,
                    )
                ],
                batch_atomicity="ATOMIC",
            ),
        )

    def start(self):
        """Starts the logger."""
        logger.info(f"Starting metric logging for session_id: {self.session_id}")
        self.started = True
        try:
            metric = Metric(
                name=MetricName.SESSION_START,
                value=1,
                timestamp=get_current_ts(),
            )
            self._post_metric(metric=metric)
        except Exception as e:
            logger.error(f"MetricLogger Failed to log session start: {str(e)}")
        return self.session_id

    def log_new_agent(
        self, agent: ConversableAgent, init_args: Dict[str, Any] = {}
    ) -> None:
        """Metric logger does not log new agent."""
        pass

    def log_function_use(
        self,
        source: Union[str, Agent],
        function: Any,
        args: Dict[str, Any],
        returns: Any,
    ) -> None:
        """
        Log a registered function(can be a tool) use from an agent or a string source.
        """
        if not self.started:
            return
        agent_name = str(source.name) if hasattr(source, "name") else source
        dimensions = {
            MetricDimension.TOOL_NAME: function.__name__,
            MetricDimension.AGENT_NAME: agent_name,
        }
        try:
            self._post_metric(
                Metric(
                    name=MetricName.TOOL_CALL,
                    value=1,
                    timestamp=get_current_ts(),
                    dimensions=dimensions,
                )
            )
        except Exception as e:
            logger.error(f"MetricLogger Failed to log tool call: {str(e)}")

    def log_chat_completion(
        self,
        invocation_id: UUID,
        client_id: int,
        wrapper_id: int,
        source: Union[str, Agent],
        request: Dict[str, Union[float, str, List[Dict[str, str]]]],
        response: Union[str, Any],
        is_cached: int,
        cost: float,
        start_time: str,
    ) -> None:
        """
        Log a chat completion.
        """
        if not self.started:
            return

        try:
            response: dict = serialize_response(response)
            if "usage" not in response or not isinstance(response["usage"], dict):
                return
            # Post usage metric
            agent_name = str(source.name) if hasattr(source, "name") else source
            model = response.get("model", "N/A")
            dimensions = {
                MetricDimension.AGENT_NAME: agent_name,
                MetricDimension.MODEL: model,
            }

            # Chat completion count
            self._post_metric(
                Metric(
                    name=MetricName.CHAT_COMPLETION,
                    value=1,
                    timestamp=get_current_ts(),
                    dimensions=dimensions,
                )
            )
            # Cost
            if cost:
                self._post_metric(
                    Metric(
                        name=MetricName.COST,
                        value=cost,
                        timestamp=get_current_ts(),
                        dimensions=dimensions,
                    )
                )
            # Usage
            for key, val in response["usage"].items():
                self._post_metric(
                    Metric(
                        name=key,
                        value=val,
                        timestamp=get_current_ts(),
                        dimensions=dimensions,
                    )
                )

        except Exception as e:
            logger.error(f"MetricLogger Failed to log chat completion: {str(e)}")

    def log_new_wrapper(
        self,
        wrapper: OpenAIWrapper,
        init_args: Dict[str, Union[LLMConfig, List[LLMConfig]]] = {},
    ) -> None:
        """Metric logger does not log new wrapper."""
        pass

    def log_new_client(self, client, wrapper, init_args):
        """Metric logger does not log new client."""
        pass

    def log_event(self, source, name, **kwargs):
        """Metric logger does not log events."""
        pass

    def get_connection(self):
        pass

    def stop(self):
        """Stops the metric logger."""
        if not self.started:
            return
        self.started = False
        try:
            metric = Metric(
                name=MetricName.SESSION_STOP,
                value=1,
                timestamp=get_current_ts(),
            )
            self._post_metric(metric=metric)
        except Exception as e:
            logger.error(f"MetricLogger Failed to log session stop: {str(e)}")
        logger.info("Metric logger stopped.")
