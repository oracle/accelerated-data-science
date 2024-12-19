#!/usr/bin/env python
# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Union
from uuid import UUID, uuid4

import oci
from autogen import Agent, ConversableAgent, OpenAIWrapper
from autogen.logger.base_logger import BaseLogger, LLMConfig
from autogen.logger.file_logger import safe_serialize
from autogen.logger.logger_utils import get_current_ts, to_dict
from oci.monitoring import MonitoringClient
from pydantic import BaseModel

import ads
import ads.common
import ads.common.oci_client

logger = logging.getLogger(__name__)


class Metric(BaseModel):
    """Represents the metric to be logged."""

    name: str
    value: float
    agent_name: str
    namespace: str
    time: str
    dimension_name: str
    dimension_value: str


class AgentMetricMonitoring(BaseLogger):
    """AutoGen logger for agent metrics."""

    def __init__(
        self,
        session_id=None,
        metric_compartment=None,
        agent_name=None,
        metric_namespace=None,
        region=None,
    ):
        self.session_id = str(session_id or uuid4())
        self.metric_compartment = metric_compartment
        auth = ads.auth.default_signer()
        signer = auth.get("signer")
        if not region:
            if not (hasattr(signer, "region") and signer.region):
                raise ValueError(
                    "Unable to determine the region for OCI monitoring service. "
                    "Please specify the region using the `region` argument."
                )
            else:
                region = signer.region
        self.monitoring_client = MonitoringClient(
            config=auth.get("config", {}),
            signer=signer,
            # Metrics should be submitted with the "telemetry-ingestion" endpoint instead.
            # See note here: https://docs.oracle.com/iaas/api/#/en/monitoring/20180401/MetricData/PostMetricData
            service_endpoint=f"https://telemetry-ingestion.{region}.oraclecloud.com",
        )
        self.agent_name = agent_name
        self.metric_namespace = metric_namespace

    def _post_metric(self, metric: Metric):
        self.monitoring_client.post_metric_data(
            post_metric_data_details=oci.monitoring.models.PostMetricDataDetails(
                metric_data=[
                    oci.monitoring.models.MetricDataDetails(
                        namespace=metric.namespace,
                        compartment_id=self.metric_compartment,
                        name=metric.name,
                        dimensions={metric.dimension_name: metric.dimension_value},
                        datapoints=[
                            oci.monitoring.models.Datapoint(
                                timestamp=datetime.strptime(
                                    metric.time.replace(" ", "T") + "Z",
                                    "%Y-%m-%dT%H:%M:%S.%fZ",
                                ),
                                value=metric.value,
                                count=1,
                            )
                        ],
                        resource_group="agent-dev-pocs",
                        metadata={"agent-name": metric.agent_name},
                    )
                ],
                batch_atomicity="ATOMIC",
            ),
        )

    def start(self):
        logger.info(f"Starting logging for session_id: {self.session_id}")
        return self.session_id

    def log_new_agent(
        self, agent: ConversableAgent, init_args: Dict[str, Any] = {}
    ) -> None:
        """
        Log a new agent instance.
        """
        logger.info(f"Event: {agent} {init_args}")

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
        try:
            log_data = {
                "source_id": id(source),
                "source_name": str(source.name) if hasattr(source, "name") else source,
                "agent_module": source.__module__,
                "agent_class": source.__class__.__name__,
                "timestamp": get_current_ts(),
                "input_args": safe_serialize(args),
                "returns": safe_serialize(returns),
            }
            metric = Metric(
                name="tool-call",
                value=1,
                dimension_value=function.__name__,
                dimension_name="tool-call",
                agent_name=self.agent_name,
                namespace=self.metric_namespace,
                time=get_current_ts(),
            )
            self._post_metric(metric=metric)
            logger.info(json.dumps(log_data))
        except Exception as e:
            self.logger.error(f"[monitoring] Failed to log event {e}")

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
        input = to_dict(request)
        output = str(response)
        if not isinstance(response, str):
            total_tokens = response.usage.total_tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            model = response.model
            metric = {
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": model,
                "cost": cost,
            }
            self._post_metric(
                metric=Metric(
                    name="token",
                    value=total_tokens,
                    agent_name=self.agent_name,
                    namespace=self.metric_namespace,
                    time=start_time,
                    dimension_name="agent",
                    dimension_value=self.agent_name,
                )
            )
            logger.info(f"Metric: {metric}")
        logger.info(f"Input: {input}")
        logger.info(f"output: {output}")

    def log_new_wrapper(
        self,
        wrapper: OpenAIWrapper,
        init_args: Dict[str, Union[LLMConfig, List[LLMConfig]]] = {},
    ) -> None:
        """
        Log a new wrapper instance.
        """
        thread_id = threading.get_ident()

        try:
            log_data = json.dumps(
                {
                    "wrapper_id": id(wrapper),
                    "session_id": self.session_id,
                    "json_state": json.dumps(init_args),
                    "timestamp": get_current_ts(),
                    "thread_id": thread_id,
                }
            )
            logger.info(log_data)
        except Exception as e:
            logger.error(f"[file_logger] Failed to log event {e}")

    def log_new_client(self, client, wrapper, init_args):
        logger.info(f"Event: {client} / {wrapper} / {init_args}")

    def log_event(self, source, name, **kwargs):
        logger.info(f"Event: {source} / {name}")

    def get_connection(self):
        pass

    def stop(self):
        logger.info("Event: Stopping...")
