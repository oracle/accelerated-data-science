from typing import List, Optional

from langchain.schema.embeddings import Embeddings
from ads.llm.langchain.plugins.base import GenerativeAiClientModel


class GenerativeAIEmbeddings(GenerativeAiClientModel, Embeddings):
    """OCI Generative AI embedding models.
    """

    model: str = "cohere.embed-english-light-v2.0"
    """Model name to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        from oci.generative_ai.models import EmbedTextDetails, OnDemandServingMode

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
