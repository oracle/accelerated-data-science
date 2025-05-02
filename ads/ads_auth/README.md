# Oracle Accelerated Data Science Auth (ADS Auth)

A standalone Python package providing authentication management for Oracle Cloud Infrastructure (OCI) clients within the Oracle Accelerated Data Science ecosystem. ADS Auth can be installed and used independently from the broader `oracle-ads` package.

---

## Features

* **Multiple Auth Methods**: Supports API Key, Resource Principal, Instance Principal, and Security Token signers.
* **Easy Integration**: Simple functions to set and retrieve signer configs for OCI SDK clients.

---

## Installation

```bash
pip install oracle-ads-auth
```

> **Note**: ADS Auth requires Python 3.9+ and depends on `pydantic` and `oci`.

---

## Quickstart

```python
import ads.auth as auth
from oci.object_storage import ObjectStorageClient

# 1. Set API Key authentication (default)
auth.set_auth(auth.AuthType.API_KEY)
client = ObjectStorageClient(**auth.default_signer())

# 2. Use Resource Principal (in OCI Functions / Data Science jobs)
auth.set_auth(auth.AuthType.RESOURCE_PRINCIPAL)
client = ObjectStorageClient(**auth.default_signer())

# 3. Override temporarily with Instance Principal
with auth.AuthContext(auth=auth.AuthType.INSTANCE_PRINCIPAL):
    client = ObjectStorageClient(**auth.default_signer())

# 4. Convenience helper
client = ObjectStorageClient(**auth.api_keys("~/.oci/config", profile="DEV"))
```

---

## API Reference

<details>
<summary><strong>auth.set_auth</strong></summary>

```python
def set_auth(
    auth: AuthType = AuthType.API_KEY,
    oci_config_location: str = DEFAULT_LOCATION,
    profile: str = DEFAULT_PROFILE,
    config: Optional[Dict[str, Any]] = None,
    signer: Optional[Any] = None,
    signer_callable: Optional[Callable[..., Any]] = None,
    signer_kwargs: Optional[Dict[str, Any]] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
```

Configure the global authentication method and parameters for subsequent client creation.

</details>

<details>
<summary><strong>auth.create_signer</strong></summary>

```python
def create_signer(...)
```

Builds an authentication dictionary (`config`, `signer`, `client_kwargs`) for any OCI SDK client.

</details>

<details>
<summary><strong>auth.default_signer</strong></summary>

```python
def default_signer(client_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
```

Retrieves the current session’s signer configuration.

</details>

<details>
<summary><strong>Convenience Wrappers</strong></summary>

* `auth.api_keys(oci_config: str|dict, profile: str, client_kwargs: dict)`
* `auth.resource_principal(client_kwargs: dict)`
* `auth.security_token(...)`

</details>

<details>
<summary><strong>AuthContext</strong></summary>

Context manager to temporarily override global auth state:

```python
with auth.AuthContext(auth=AuthType.RESOURCE_PRINCIPAL):
    # operations use resource principal
```

</details>


## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md)


## Security

Consult the security guide [SECURITY.md](https://github.com/oracle/accelerated-data-science/blob/main/SECURITY.md) for our responsible security vulnerability disclosure process.

## License

Copyright (c) 2020, 2025 Oracle and/or its affiliates. Licensed under the [Universal Permissive License v1.0](https://oss.oracle.com/licenses/upl/)
