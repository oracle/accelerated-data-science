#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64
import json
import os
from copy import deepcopy
from langchain.load import dumpd
from langchain.chains.loading import load_chain_from_config
from langchain.vectorstores import FAISS, OpenSearchVectorSearch


class OpenSearchVectorDBSerializer:
    """
    Serializer for OpenSearchVectorSearch class
    """

    @staticmethod
    def type():
        return OpenSearchVectorSearch.__name__

    @staticmethod
    def load(config: dict, **kwargs):
        from ads.llm.serialize import load

        config["kwargs"]["embedding_function"] = load(
            config["kwargs"]["embedding_function"], **kwargs
        )
        return OpenSearchVectorSearch(
            **config["kwargs"],
            http_auth=(
                os.environ.get("OCI_OPENSEARCH_USERNAME", None),
                os.environ.get("OCI_OPENSEARCH_PASSWORD", None),
            ),
            verify_certs=True
            if os.environ.get("OCI_OPENSEARCH_VERIFY_CERTS", None) == "True"
            else False,
            ca_certs=os.environ.get("OCI_OPENSEARCH_CA_CERTS", None),
        )

    @staticmethod
    def save(obj):
        from ads.llm.serialize import dump
        from opensearchpy.client import OpenSearch

        serialized = dumpd(obj)
        serialized["type"] = "constructor"
        serialized["_type"] = OpenSearchVectorDBSerializer.type()
        kwargs = {}
        for component_name, component in obj.__dict__.items():
            if component_name == "client":
                if isinstance(component, OpenSearch):
                    client_info = component.transport.hosts[0]
                    opensearch_url = (
                        f"https://{client_info['host']}:{client_info['port']}"
                    )
                    kwargs.update({"opensearch_url": opensearch_url})
                else:
                    raise NotImplementedError("Only support OpenSearch client.")
                continue
            kwargs[component_name] = dump(component)
        serialized["kwargs"] = kwargs
        return serialized


class FaissSerializer:
    """
    Serializer for OpenSearchVectorSearch class
    """

    @staticmethod
    def type():
        return FAISS.__name__

    @staticmethod
    def load(config: dict, **kwargs):
        from ads.llm.serialize import load

        embedding_function = load(config["embedding_function"], **kwargs)
        decoded_pkl = base64.b64decode(json.loads(config["vectordb"]))
        return FAISS.deserialize_from_bytes(
            embeddings=embedding_function, serialized=decoded_pkl
        )  # Load the index

    @staticmethod
    def save(obj):
        from ads.llm.serialize import dump

        serialized = {}
        serialized["_type"] = FaissSerializer.type()
        pkl = obj.serialize_to_bytes()
        # Encoding bytes to a base64 string
        encoded_pkl = base64.b64encode(pkl).decode("utf-8")
        # Serializing the base64 string
        serialized["vectordb"] = json.dumps(encoded_pkl)
        serialized["embedding_function"] = dump(obj.__dict__["embedding_function"])
        return serialized


class RetrievalQASerializer:
    """
    Serializer for RetrieverQA class
    """

    # Mapping class to vector store serialization functions
    vectordb_serialization = {
        "OpenSearchVectorSearch": OpenSearchVectorDBSerializer,
        "FAISS": FaissSerializer,
    }

    @staticmethod
    def type():
        return "retrieval_qa"

    @staticmethod
    def load(config: dict, **kwargs):
        config_param = deepcopy(config)
        retriever_kwargs = config_param.pop("retriever_kwargs")
        vectordb_serializer = RetrievalQASerializer.vectordb_serialization[
            config_param["vectordb"]["class"]
        ]
        vectordb = vectordb_serializer.load(config_param.pop("vectordb"), **kwargs)
        retriever = vectordb.as_retriever(**retriever_kwargs)
        return load_chain_from_config(config=config_param, retriever=retriever)

    @staticmethod
    def save(obj):
        serialized = obj.dict()
        retriever_kwargs = {}
        for key, val in obj.retriever.__dict__.items():
            if key not in ["tags", "metadata", "vectorstore"]:
                retriever_kwargs[key] = val
        serialized["retriever_kwargs"] = retriever_kwargs
        serialized["vectordb"] = {"class": obj.retriever.vectorstore.__class__.__name__}

        vectordb_serializer = RetrievalQASerializer.vectordb_serialization[
            serialized["vectordb"]["class"]
        ]
        serialized["vectordb"].update(
            vectordb_serializer.save(obj.retriever.vectorstore)
        )

        if (
            serialized["vectordb"]["class"]
            not in RetrievalQASerializer.vectordb_serialization
        ):
            raise NotImplementedError(
                f"VectorDBSerializer for {serialized['vectordb']['class']} is not implemented."
            )
        return serialized
