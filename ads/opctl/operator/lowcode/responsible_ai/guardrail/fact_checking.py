from nemoguardrails.kb.kb import KnowledgeBase
from .llm import OCIEndpoint
import oci
from nemoguardrails.actions.retrieve_relevant_chunks import retrieve_relevant_chunks
from typing import List


async def check_fact(question, response, paths: List[str]):
    oci_config = oci.config.from_file()

    endpoint = OCIEndpoint(
        compartment_id=oci_config["compartment_id"],
        config=oci_config,
        service_endpoint="https://generativeai-dev.aiservice.us-chicago-1.oci.oraclecloud.com",
    )
    documents = []
    for path in paths:
        with open(path, "r") as f:
            documents.append(f.read())

    kb = KnowledgeBase(
        documents=documents, embedding_model="all-MiniLM-L6-v2"
    )
    kb.init()
    kb.build()
    context = {"last_user_message": question}
    retrieved = retrieve_relevant_chunks(context, kb)
    result = await retrieved
    evidence = result.return_value
    prompt = f"""You are given a task to identify if the hypothesis is grounded and entailed to the evidence.
      You will only use the contents of the evidence and not rely on external knowledge.
      Answer with yes/no. "evidence": { evidence } "hypothesis": { response } "entails":"""
    return "yes" in endpoint.generate(prompt=prompt)