---
name: aqua-metrics
description: Set up Prometheus and Grafana monitoring for AQUA vLLM model deployments on OCI. Covers the signing proxy, container registry setup, OCI Container Instance deployment, and PromQL dashboards. Triggered when user wants to monitor LLM deployments, view TTFT/latency/throughput metrics, or set up observability for AQUA.
user-invocable: true
disable-model-invocation: false
---

# AQUA Deployment Metrics Monitoring

Monitor vLLM model deployments with Prometheus + Grafana hosted on an OCI Container Instance. The monitoring stack consists of:

- **Signing Proxy** — handles OCI IAM auth when scraping the `/metrics` endpoint
- **Prometheus** — scrapes metrics every 5s, stores time series
- **Grafana** — visualizes dashboards from Prometheus data

## Available Metrics (vLLM Prometheus)

All standard vLLM Prometheus metrics are available:

| Metric | Description |
|---|---|
| `vllm:time_to_first_token_seconds` | TTFT histogram |
| `vllm:inter_token_latency_seconds` | ITL histogram |
| `vllm:e2e_request_latency_seconds` | End-to-end request latency |
| `vllm:num_requests_running` | Concurrent requests in flight |
| `vllm:num_requests_waiting` | Requests queued |
| `vllm:gpu_cache_usage_perc` | KV cache utilization |
| `vllm:num_tokens_prompt` | Prompt token count |
| `vllm:num_tokens_generation` | Generation token count |
| `vllm:request_success_total` | Successful request count |

Full list: https://docs.vllm.ai/en/latest/design/metrics/

## Architecture

```
AQUA Model Deployment
  └── /predict/metrics endpoint (requires OCI IAM signature)
           ↑
    Signing Proxy :8080
    (resource_principal auth)
           ↑
    Prometheus :9090
    (scrapes localhost:8080 every 5s)
           ↑
    Grafana :3000
    (visualizes from localhost:9090)
           ↑
    User browser (public IP of Container Instance)
```

## Step 1: Clone the Monitoring Stack

```bash
git clone https://github.com/oracle-samples/oci-data-science-ai-samples.git
cd oci-data-science-ai-samples/ai-quick-actions/aqua_metrics
```

The directory contains:
- `signing_proxy/` — OCI-aware auth proxy (Dockerfile)
- `prometheus/` — Prometheus config + Dockerfile
- `grafana/` — Grafana Dockerfile

## Step 2: Build and Push Images to OCIR

Replace `<registry-domain>` with your region's OCIR endpoint (e.g., `iad.ocir.io`) and `<tenancy-namespace>` with your tenancy namespace.

### Signing Proxy

```bash
cd signing_proxy
docker build --no-cache -t signing_proxy .
docker tag signing_proxy <registry-domain>/<tenancy-namespace>/signing_proxy
docker push <registry-domain>/<tenancy-namespace>/signing_proxy:latest
```

### Prometheus

The `prometheus/prometheus.yml` is preconfigured to scrape `localhost:8080` (the proxy):

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 30s
scrape_configs:
  - job_name: AQUA
    static_configs:
      - targets:
          - 'localhost:8080'
```

```bash
cd ../prometheus
docker build --no-cache -t prometheus .
docker tag prometheus <registry-domain>/<tenancy-namespace>/prom/prometheus
docker push <registry-domain>/<tenancy-namespace>/prom/prometheus:latest
```

### Grafana

```bash
cd ../grafana
docker build --no-cache -t grafana .
docker tag grafana <registry-domain>/<tenancy-namespace>/grafana/grafana
docker push <registry-domain>/<tenancy-namespace>/grafana/grafana:latest
```

> Alternative: pull `grafana/grafana` directly from `docker.io` on the Container Instance — no build needed.

## Step 3: Create the OCI Container Instance

In the OCI Console: **Developer Services** → **Containers & Artifacts** → **Container Instances** → **Create container instance**

### Network Configuration

- Create or select a VCN with a public or private regional subnet
- Security list must allow **ingress** on ports: `8080`, `9090`, `3000`
- Security list must allow **egress** to the model deployment endpoint
- Check **Assign a public IPv4 address** for external Grafana access

### Configure Three Containers

Add each container from OCIR:

**signing_proxy:**
- Image: `<registry-domain>/<tenancy-namespace>/signing_proxy:latest`
- Environment variable: `TARGET = <model-deployment-url>/predict/metrics`
  - Format: `https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict/metrics`

**prometheus:**
- Image: `<registry-domain>/<tenancy-namespace>/prom/prometheus:latest`
- No extra env vars needed (config is baked in)

**grafana:**
- Image: `<registry-domain>/<tenancy-namespace>/grafana/grafana:latest`
- Environment variable: `PORT = 3000`

## Step 4: Configure Grafana

Once the Container Instance is active:

1. Open `http://<container-instance-public-ip>:3000`
2. Log in with `admin / admin` (change on first login)
3. Go to **Configuration** → **Data Sources** → **Add data source**
4. Select **Prometheus**
5. URL: `http://localhost:9090`
6. Click **Save & Test** — should show "Data source is working"

### Example PromQL Queries

```promql
# TTFT p50 / p95 / p99
histogram_quantile(0.5, rate(vllm:time_to_first_token_seconds_bucket[1m]))
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[1m]))
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[1m]))

# Requests per second
rate(vllm:request_success_total[1m])

# KV cache utilization
vllm:gpu_cache_usage_perc

# Active requests
vllm:num_requests_running
vllm:num_requests_waiting

# Tokens per second (generation)
rate(vllm:num_tokens_generation[1m])
```

For Grafana dashboard templates, see: https://grafana.com/docs/grafana/latest/getting-started/build-first-dashboard/

## Exposing the Metrics Endpoint

The AQUA model deployment exposes Prometheus metrics at:
```
<deployment-url>/predict/metrics
```

The signing proxy handles OCI IAM signatures via `resource_principal` so Prometheus can scrape without managing OCI credentials directly.

## Key Source Files

- `oracle-samples/oci-data-science-ai-samples` — `ai-quick-actions/aqua_metrics/`
- `ads/aqua/modeldeployment/deployment.py` — deployment endpoint management
