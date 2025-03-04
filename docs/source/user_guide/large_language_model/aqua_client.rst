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
