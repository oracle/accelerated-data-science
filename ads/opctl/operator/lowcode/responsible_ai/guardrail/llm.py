import oci
import requests
from generative_ai_service_bmc_python_client import GenerativeAIClient, models


class LLMEndpoint:
    def generate(
        self,
        prompt,
        model="command-nightly",
        max_tokens=1400,
        temperature=0.1,
        p=0.9,
        k=0,
        **kwargs,
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
        **kwargs,
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
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def embed(self, texts, model="embed-english-light-v2.0"):
        raise NotImplementedError


class NotAuthorizedError(oci.exceptions.ServiceError):
    pass


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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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


class MDEndpoint(LLMEndpoint):
    def __init__(self, endpoint, config=None, signer=None) -> None:
        self.endpoint = endpoint
        self.config = config
        self.signer = signer
        super().__init__()

    def generate(
        self, prompt, model=None, max_tokens=512, temperature=0.1, p=0.9, k=0, **kwargs
    ) -> str:
        parameters = {
            "best_of": 1,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "return_full_text": False,
            "watermark": True,
            # "seed": 42,
            "temperature": temperature,
            "top_p": p,
            **kwargs,
        }
        if k > 0:
            parameters["top_k"] = k
        body = {
            "inputs": prompt,
            "parameters": parameters,
        }
        model = model if model else self.endpoint
        response = requests.post(model, json=body, auth=self.signer, timeout=300)
        data = response.json()
        if data.get("code") == "NotAuthorizedOrNotFound":
            data["status"] = response.status_code
            data["headers"] = response.headers
            raise NotAuthorizedError(**data)
        return str(data.get("generated_text", data))
    
    def batch_generate(
        self,
        prompt,
        model=None,
        max_tokens=512,
        temperature=0.1,
        p=0.9,
        k=0, 
        num_generations=1,
        **kwargs):
        res = []
        for i in range(num_generations):
            res.append(self.generate(prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            p=p,
            k=k, 
            **kwargs))
        return res


class LLMManager:
    def __init__(self) -> None:
        self.endpoints = {}

    @staticmethod
    def get_oci_auth(profile):
        oci_config = oci.config.from_file(profile_name=profile)
        if "security_token_file" in oci_config and "key_file" in oci_config:
            token_file = oci_config["security_token_file"]
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read()
            private_key = oci.signer.load_private_key_from_file(oci_config["key_file"])
            signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
            oci_auth = {"config": oci_config, "signer": signer}
        else:
            oci_auth = {"config": oci_config}
        return oci_auth

    def get(self, name, **kwargs):
        if name not in self.endpoints:
            self.endpoints[name] = self.init_endpoint(name, **kwargs)
        return self.endpoints[name]

    def init_endpoint(self, name, profile, endpoint) -> None:
        if name == "cohere":
            oci_config = oci.config.from_file(profile_name=profile)
            return OCIEndpoint(
                compartment_id=oci_config["tenancy"],
                service_endpoint=endpoint,
                config=oci_config,
            )
        elif name in ["llama7b", "llama13b"]:
            oci_auth = self.get_oci_auth(profile)
            return MDEndpoint(endpoint=endpoint, **oci_auth)
        else:
            raise NotImplementedError(f"Model {name} not implemented.")
