AI Quick Actions HTTP Client
****************************

.. versionadded:: 2.13.0

The AI Quick Actions client is a centralized, reusable component for interacting with the OCI Model Deployment service.

**Implementation Highlights:**

- Offers both synchronous (Client) and asynchronous (AsyncClient)
- Integrates with OCI Authentication patterns

Authentication
==============

The AI Quick Actions client supports the same authentication methods as other OCI services, including API Key, session token, instance principal, and resource principal. For additional details, please refer to the `authentication guide <https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html>`_. Ensure you have the necessary `access policies <https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm>`_ to connect to the OCI Data Science Model Deployment endpoint.

Usage
=====

Sync Usage
----------

**Text Completion**

.. code-block:: python3

    from ads.aqua import Client
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = Client(endpoint="https://<MD_OCID>/predict")
    response = client.generate(
        prompt="Tell me a joke",
        payload={"model": "odsc-llm"},
        stream=False,
    )
    print(response)

**Chat Completion**

.. code-block:: python3

    from ads.aqua import Client
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = Client(endpoint="https://<MD_OCID>/predict")
    response = client.chat(
        messages=[{"role": "user", "content": "Tell me a joke."}],
        payload={"model": "odsc-llm"},
        stream=False,
    )
    print(response)

**Streaming**

.. code-block:: python3

    from ads.aqua import Client
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = Client(endpoint="https://<MD_OCID>/predict")
    response = client.chat(
        messages=[{"role": "user", "content": "Tell me a joke."}],
        payload={"model": "odsc-llm"},
        stream=True,
    )

    for chunk in response:
        print(chunk)

**Embedding**

.. code-block:: python3

    from ads.aqua import Client
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = Client(endpoint="https://<MD_OCID>/predict")
    response = client.embeddings(
        input=["one", "two"]
    )
    print(response)


Async Usage
-----------

The following examples demonstrate how to perform the same operations using the asynchronous client with Python's async/await syntax.

**Text Completion**

.. code-block:: python3

    from ads.aqua import AsyncClient
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = AsyncClient(endpoint="https://<MD_OCID>/predict")
    response = await client.generate(
        prompt="Tell me a joke",
        payload={"model": "odsc-llm"},
        stream=False,
    )
    print(response)

**Streaming**

.. code-block:: python3

    from ads.aqua import AsyncClient
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = AsyncClient(endpoint="https://<MD_OCID>/predict")
    async for chunk in await client.generate(
        prompt="Tell me a joke",
        payload={"model": "odsc-llm"},
        stream=True,
    ):
        print(chunk)

**Embedding**

.. code-block:: python3

    from ads.aqua import AsyncClient
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = AsyncClient(endpoint="https://<MD_OCID>/predict")
    response = await client.embeddings(
        input=["one", "two"]
    )
    print(response)


HTTPX Client Integration with OCI Authentication
================================================

.. versionadded:: 2.13.1

The latest client release now includes streamlined support for OCI authentication with HTTPX. Our helper functions for creating synchronous and asynchronous HTTPX clients automatically configure authentication based on your default settings. Additionally, you can pass extra keyword arguments to further customize the HTTPX client (e.g., timeouts, proxies, etc.), making it fully compatible with OCI Model Deployment service and third-party libraries (e.g., the OpenAI client).

Usage
-----

**Synchronous HTTPX Client**

.. code-block:: python3

    import ads
    import ads.aqua

    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = ads.aqua.get_httpx_client(timeout=10.0)

    response = client.post(
        url="https://<MD_OCID>/predict",
        json={
            "model": "odsc-llm",
            "prompt": "Tell me a joke."
        },
    )

    response.raise_for_status()
    json_response = response.json()

**Asynchronous HTTPX Client**

.. code-block:: python3

    import ads
    import ads.aqua

    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    async_client = client = ads.aqua.get_async_httpx_client(timeout=10.0)


Aqua OpenAI Client
==================

.. versionadded:: 2.13.4

The **AquaOpenAI** and **AsyncAquaOpenAI** clients extend the official OpenAI Python SDK to support OCI-based model deployments. They automatically patch request headers and normalize URL paths based on the deployment OCID, ensuring that API calls are sent in the proper format.

Requirements
------------
To use these clients, you must have the ``openai-python`` package installed. This package is an optional dependency. If it is not installed, you will receive an informative error when attempting to instantiate one of these clients. To install the package, run:

.. code-block:: bash

   pip install openai


Usage
-----
Both synchronous and asynchronous versions are available.

**Synchronous Client**

The synchronous client, ``AquaOpenAI``, extends the OpenAI client. If no HTTP client is provided, it will automatically create one using ``ads.aqua.get_httpx_client()``.

.. code-block:: python

    import ads
    from ads.aqua.client.openai_client import AquaOpenAI
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")

    client = AquaOpenAI(
        base_url="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<OCID>/predict",
    )

    response = client.chat.completions.create(
        model="odsc-llm",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke.",
            }
        ],
        # stream=True, # enable for streaming
    )

    print(response)


**Asynchronous Client**

The asynchronous client, ``AsyncAquaOpenAI``, extends the AsyncOpenAI client. If no async HTTP client is provided, it will automatically create one using ``ads.aqua.get_async_httpx_client()``.

.. code-block:: python

    import ads
    import asyncio
    import nest_asyncio
    from ads.aqua.client.openai_client import AsyncAquaOpenAI

    ads.set_auth(auth="security_token")

    async def test_async() -> None:
        client_async = AsyncAquaOpenAI(
            base_url="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<OCID>/predict",
        )
        response = await client_async.chat.completions.create(
            model="odsc-llm",
            messages=[{"role": "user", "content": "Tell me a long joke"}],
            stream=True
        )
        async for event in response:
            print(event)

    asyncio.run(test_async())
