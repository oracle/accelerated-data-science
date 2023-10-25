from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env
from ads.common.auth import default_signer
from ads.config import COMPARTMENT_OCID


class GenerativeAIEmbeddings(BaseModel, Embeddings):
    """OCI Generative AI embedding models.
    """

    client: Any  #: :meta private:
    """OCI Generative AI client."""

    compartment_id: str

    model: str = "cohere.embed-english-light-v2.0"
    """Model name to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    def __init__(self, **kwargs):
        """The embedding model with OCI Generative AI client.

        In addition to the fields of the model, users may also passed in extra keyword arguments,
        for example, ``service_endpoint``. The extra keyword arguments will be use to initialize the OCI client.
        Alternatively, user may also initialize the client outside of this model and pass it in directly.

        The ADS ``default_signer()`` is used for authentication.
        You may configure it through ``ads.set_auth()``.

        Example:
        .. code-block:: python
            import ads
            from ads.llm import GenerativeAIEmbeddings

            ads.set_auth(auth="security_token", profile="...")

            oci_embeddings = GenerativeAIEmbeddings(
                compartment_id="ocid1.compartment.oc1...",
                service_endpoint="..."
            )

        Raises
        ------
        ImportError
            If the OCI SDK installed does not support generative AI service.
        ValueError
            If the compartment_id is not specified or cannot be obtained from the environment.
        """
        try:
            # Import the GenerativeAIClient here so that there will be no error when user import ads.llm
            # and the install OCI SDK does not support generative AI service yet.
            from oci.generative_ai import GenerativeAiClient
        except ImportError as ex:
            raise ImportError(
                "Could not import GenerativeAIClient from oci. "
                "The OCI SDK installed does not support generative AI service."
            ) from ex
        # Initialize client only if user does not pass in client.
        # Users may choose to initialize the OCI client by themselves and pass it into this model.
        if not kwargs.get("client"):
            client_kwargs = default_signer()
            # Extra key-value pairs that are not fields will be passed into the GenerativeAiClient()
            for key in list(kwargs.keys()):
                if key not in self.__fields__:
                    client_kwargs[key] = kwargs.pop(key)
            kwargs["client"] = GenerativeAiClient(**client_kwargs)
        # Set default compartment ID
        if "compartment_id" not in kwargs:
            kwargs["compartment_id"] = COMPARTMENT_OCID
        if not kwargs["compartment_id"]:
            raise ValueError("compartment_id is required.")
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            # Import the GenerativeAIClient here so that there will be no error when user import ads.llm
            # and the install OCI SDK does not support generative AI service yet.
            from oci.generative_ai.models import EmbedTextDetails, OnDemandServingMode

        except ImportError as ex:
            raise ImportError(
                "Could not import GenerativeAIClient from oci. "
                "The OCI SDK installed does not support generative AI service."
            ) from ex
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
