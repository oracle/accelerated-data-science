from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.actions.retrieve_relevant_chunks import retrieve_relevant_chunks
from .base import BaseGuardRail
from typing import List, Union
from ...utils import init_endpoint, apply_filter
import asyncio
import pandas as pd
import oci


class FactChecking(BaseGuardRail):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        # endpoint used to check fact, which is usually the llm used to get response from prompt

    def load(self,):
        self.knowledge = self.config.get("evaluation", {}).get("load_args", {}).get("knowledge", [])
        self.embedding_model = self.config.get("evaluation", {}).get("load_args", {}).get("embedding_model", "all-MiniLM-L6-v2")
        self.custom_msg = self.config.get("action", {}).get("custom_msg", "I'm sorry, I don't know the answer to that question.")
        documents = []
        for path in self.knowledge:
            with open(path, "r") as f:
                documents.append(f.read())
        self.kb = KnowledgeBase(documents=documents, embedding_model=self.embedding_model)
        self.kb.init()
        self.kb.build()
        return self

    async def _compute(self, prompt: str, response: str, **kwargs):
        context = {"last_user_message": prompt}
        retrieved = retrieve_relevant_chunks(context, self.kb)
        result = await retrieved
        evidence = result.return_value
        instruction = f"""You are given a task to identify if the hypothesis is grounded and entailed to the evidence.
        You will only use the contents of the evidence and not rely on external knowledge.
        Answer with yes/no. "evidence": { evidence } "hypothesis": { response } "entails":"""

        endpoint = kwargs.get("endpoint") or init_endpoint(name=self.config.get("evaluation", {}).get("compute_args", {}).get("llm_model"))
        return 1 if "yes" in endpoint.generate(prompt=instruction) else 0
    
    async def _batch_compute(self, prompt: str, responses: List, **kwargs):
        loop = asyncio.get_event_loop()
        tasks = []
        for response in responses:
            task = loop.create_task(self._compute(prompt=prompt, response=response, **kwargs))
            tasks.append(task)
        return await asyncio.gather(*tasks)
    
    def compute(self, predictions: str, prompt: str, **kwargs):
        return {"fact_checking": pd.DataFrame(asyncio.run(self._batch_compute(prompt=prompt, responses=predictions, **kwargs)), columns=['fact_checking'])}
    
    def apply_filter(self, score: pd.DataFrame, direction: str=">="):
        return apply_filter(score=score,
                            threshold=1,
                            direction=direction)

    @property
    def description(self):
        return (
            "You selected Fact Checking module to validate the response. "
            + "Fact checking compares the responses against text retrieved from a knowledge base to see if the responses are accurate."
        )

    @property
    def homepage(self):
        return "https://github.com/NVIDIA/NeMo-Guardrails/blob/main/examples/grounding_rail/README.md"


def main():
    ## code to use llama2 endpoint
    import oci
    from ...utils import authenticate_with_security_token, get_oci_auth
    from ..llm import LLMEndpoint, MDEndpoint, OCIEndpoint
    authenticate_with_security_token(
                "custboat")
    oci_config = get_oci_auth("custboat")
    
    print(oci_config)
    endpoint = MDEndpoint(config=oci_config, endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.iad.amaaaaaay75uckqay7so6w2bpwreqxisognml72kdqi4qcjdtnpfykh4xtsq/predict")

    # code to use cohere endpoint
    # oci_config = oci.config.from_file()
    # from ..llm import LLMEndpoint, MDEndpoint, OCIEndpoint
    # endpoint = OCIEndpoint(
    #     compartment_id=oci_config["compartment_id"],
    #     config=oci_config,
    #     service_endpoint="https://generativeai-dev.aiservice.us-chicago-1.oci.oraclecloud.com",
    # )

    config = {'name': 'FactChecking', 'evaluation': {'load_args': {'knowledge': ['knowledge1.txt'], 'embedding_model': 'all-MiniLM-L6-v2'}}, 'action': {'custom_msg': "I'm sorry, I don't know the answer to that question."}}

    prompt = "what is the unemployment rate in March 2023?"
    response = """The unemployment rate in March 2023 is 3.7%, which is the lowest rate since May 2000.

    The labor market has shown resilience in recent years, with job gains occurring across a wide range of industries. However, there are still concerns about the impact of automation and artificial intelligence on jobs, as well as the ongoing COVID-19 pandemic.

    The unemployment rate for different demographic groups in March 2023 is as follows:

    Adult men: 3.4%
    Adult women: 3.9%
    Young adults (ages 18-29): 6.5%
    Older adults (ages 55-64): 3.1%
    Persons with a disability: 6.1%
    Overall, the labor market in March 2023 is strong, with low unemployment and continued job gains across a wide range of industries. However, there are still concerns about the impact of automation and artificial intelligence on jobs, as well as the ongoing COVID-19 pandemic."""
    fact = FactChecking(name="fact_checking", config=config) 
    print(fact.compute(prompt, response, endpoint=endpoint))


if __name__ == "__main__":
    # python -m ads.opctl.operator.lowcode.responsible_ai.guardrail.rails.fact_checking
    main()