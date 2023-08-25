import time
import os
from generative_ai_service_bmc_python_client import GenerativeAIClient, models
import itertools
from .utils import calculate_similarity, get_centroid


class LLMEndpoint:
    def generate(
        self,
        prompt,
        model="command-nightly",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        **kwargs
    ) -> str:
        raise NotImplementedError

    def batch_generate(
        self,
        prompt,
        model="command",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        num_generations=1,
        **kwargs
    ) -> list:
        raise NotImplementedError

    def summarize(
        self,
        text,
        model="summarize-xlarge",
        length="AUTO",
        temperature=0.1,
        format="PARAGRAPH",
        extractiveness="MEDIUM",
        additional_command="",
        **kwargs
    ) -> str:
        raise NotImplementedError

    def embed(self, texts, model="embed-english-light-v2.0"):
        raise NotImplementedError

    def generate_centroid_generation(
        self,
        prompt,
        num_generations=5,
        generate_model="command",
        embed_model="embed-english-light-v2.0",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        **kwargs
    ):
        raise NotImplementedError


class OCIEndpoint(LLMEndpoint):
    def __init__(self, compartment_id, config=None, signer=None, **kwargs) -> None:
        super().__init__()
        self.compartment_id = compartment_id
        if config:
            self.client = GenerativeAIClient(config=config, **kwargs)
        elif signer:
            self.client = GenerativeAIClient(config=None, signer=signer, **kwargs)
        else:
            raise ValueError("Please specify OCI config or signer.")

    def generate(
        self,
        prompt,
        model="command",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        **kwargs
    ):
        details = models.GenerateTextDetails(
            compartment_id=self.compartment_id,
            prompts=[prompt],
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=p,
            top_k=k,
            **kwargs,
        )
        return self.client.generate_text(details).data.generated_texts[0][0].text

    def batch_generate(
        self,
        prompt,
        model="command",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        num_generations=1,
        **kwargs
    ) -> list:
        details = models.GenerateTextDetails(
            compartment_id=self.compartment_id,
            prompts=[prompt],
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=p,
            top_k=k,
            num_generations=num_generations,
            **kwargs,
        )
        return [
            gen.text
            for gen in self.client.generate_text(details).data.generated_texts[0]
        ]

    def summarize(
        self,
        text,
        model="summarize-xlarge",
        length="AUTO",
        temperature=0.1,
        format="PARAGRAPH",
        extractiveness="MEDIUM",
        additional_command="",
        **kwargs
    ):
        details = models.SummarizeTextDetails(
            compartment_id=self.compartment_id,
            input=text,
            model_id=model,
            length=length,
            temperature=temperature,
            format=format,
            extractiveness=extractiveness,
            additional_command=additional_command,
            **kwargs,
        )
        return self.client.summarize_text(details).data.summary

    def embed(self, texts, model="embed-english-light-v2.0"):
        details = models.EmbedTextDetails(
            compartment_id=self.compartment_id, inputs=texts, model_id=model
        )
        return self.client.embed_text(details).data.embeddings

    def generate_centroid_generation(
        self,
        prompt,
        num_generations=5,
        generate_model="command",
        embed_model="embed-english-light-v2.0",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        **kwargs
    ):
        # print("temperature:", temperature)
        generations = self.batch_generate(
            prompt,
            num_generations=num_generations,
            model=generate_model,
            max_tokens=max_tokens,
            temperature=temperature,
            p=p,
            k=k,
            **kwargs,
        )
        # print("generations: ", generations)
        embeddings = self.embed(texts=generations, model=embed_model)
        pairs = list(itertools.combinations(range(len(generations)), 2))
        # similarities
        similarities = {}
        for i, j in pairs:
            similarities[(i, j)] = calculate_similarity(embeddings[i], embeddings[j])
        centroid = get_centroid(similarities)
        return generations[centroid]
