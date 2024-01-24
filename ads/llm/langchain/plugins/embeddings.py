#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional
from langchain.load.serializable import Serializable
from langchain.schema.embeddings import Embeddings
from ads.llm.langchain.plugins.base import GenerativeAiClientModel


class GenerativeAIEmbeddings(GenerativeAiClientModel, Embeddings, Serializable):
    """OCI Generative AI embedding models."""

    model: str = "cohere.embed-english-light-v2.0"
    """Model name to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the LangChain object."""
        return ["ads", "llm"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """This class can be serialized with default LangChain serialization."""
        return True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        from oci.generative_ai_inference.models import (
            EmbedTextDetails,
            OnDemandServingMode,
        )

        details = EmbedTextDetails(
            compartment_id=self.compartment_id,
            inputs=texts,
            serving_mode=OnDemandServingMode(model_id=self.model),
            truncate=self.truncate,
        )
        embeddings = self.client.embed_text(details).data.embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single string.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
