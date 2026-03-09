---
name: aqua-evaluation
description: Evaluate LLM model quality using BERTScore, ROUGE, Perplexity, and Text Readability metrics on OCI AI Quick Actions (AQUA). Covers dataset preparation, evaluation job creation, and report interpretation. Triggered when user wants to evaluate or benchmark a model.
user-invocable: true
disable-model-invocation: false
---

# AQUA Model Evaluation

Use this skill when the user wants to evaluate LLM models on OCI Data Science using AI Quick Actions.

## Supported Metrics

| Metric | Description | Best For |
|---|---|---|
| **BERTScore** | Embedding-based semantic similarity (precision, recall, F1) | General text quality, aligns well with human judgement |
| **ROUGE** | N-gram overlap between generated and reference text | Summarization tasks |
| **Perplexity** | How well the model predicts the text | Language modeling quality |
| **Text Readability** | Reading level / complexity of generated text | Content accessibility |

## Dataset Format

JSONL format with required `prompt` and `completion` keys, optional `category`:

```jsonl
{"prompt": "Summarize this dialog:\nAmanda: I baked cookies...", "completion": "Amanda baked cookies and will bring some for Jerry tomorrow."}
{"prompt": "Translate to French: Hello world", "completion": "Bonjour le monde", "category": "translation"}
{"prompt": "What is 2+2?", "completion": "4", "category": "math"}
```

The `category` field dimensions evaluation metrics in the report (e.g., see accuracy per category).
When omitted, defaults to `"_"` (unknown).

Official sample datasets (10 prompts each, math + logic categories):

| File | Use Case |
|---|---|
| `examples/evaluation-sample-no-sys-message.jsonl` | Llama-style prompts without system message |
| `examples/evaluation-sample-with-sys-message.jsonl` | Llama-style prompts with `<<SYS>>` system message |

## Python SDK Usage

### Import

```python
from ads.aqua.evaluation import AquaEvaluationApp
eval_app = AquaEvaluationApp()
```

### Create Evaluation

```python
from ads.aqua.evaluation.entities import CreateAquaEvaluationDetails

details = CreateAquaEvaluationDetails(
    evaluation_source_id="ocid1.datasciencemodeldeployment.oc1.iad.xxx",  # Deployment OCID
    evaluation_name="llama-3.1-8b-eval-bertscore",
    dataset_path="oci://my-bucket@my-namespace/datasets/eval_data.jsonl",
    report_path="oci://my-bucket@my-namespace/eval-reports/",
    model_parameters={
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    shape_name="VM.Standard.E4.Flex",
    block_storage_size=50,
    compartment_id="ocid1.compartment.oc1..xxx",
    project_id="ocid1.datascienceproject.oc1.iad.xxx",
    log_group_id="ocid1.loggroup.oc1.iad.xxx",
    log_id="ocid1.log.oc1.iad.xxx",
    metrics=[
        {"name": "bertscore"},
        {"name": "rouge"},
    ],
)
evaluation = eval_app.create(create_evaluation_details=details)
print(f"Evaluation: {evaluation.id} | State: {evaluation.lifecycle_state}")
```

### Evaluate a Model (not deployment) Directly

You can pass a model OCID instead of a deployment OCID:
```python
details = CreateAquaEvaluationDetails(
    evaluation_source_id="ocid1.datasciencemodel.oc1.iad.xxx",  # Model OCID
    # ... rest of params same as above
)
```

### Evaluate Stacked/Multi-Model Deployment

For stacked or multi-model deployments, specify which model to evaluate:
```python
details = CreateAquaEvaluationDetails(
    evaluation_source_id="ocid1.datasciencemodeldeployment.oc1.iad.xxx",
    model_parameters={
        "max_tokens": 500,
        "temperature": 0.7,
        "model": "llama-3.1-8b-customer-support",  # Target specific model in deployment
    },
    # ... rest of params
)
```

### With Experiment Tracking

```python
details = CreateAquaEvaluationDetails(
    evaluation_source_id="ocid1.datasciencemodeldeployment.oc1.iad.xxx",
    evaluation_name="llama-eval-v2",
    dataset_path="oci://my-bucket@my-namespace/datasets/eval_data.jsonl",
    report_path="oci://my-bucket@my-namespace/eval-reports/",
    model_parameters={"max_tokens": 500, "temperature": 0.7},
    shape_name="VM.Standard.E4.Flex",
    block_storage_size=50,
    experiment_name="llama-evaluations",  # Groups evaluations together
    experiment_description="Llama 3.1 evaluation experiments",
    metrics=[{"name": "bertscore"}, {"name": "rouge"}],
)
```

### List Evaluations

```python
evaluations = eval_app.list(compartment_id="ocid1.compartment.oc1..xxx")
for e in evaluations:
    print(f"{e.display_name} | {e.lifecycle_state}")
```

### Get Evaluation Details

```python
evaluation = eval_app.get(eval_id="ocid1.datasciencemodel.oc1.iad.xxx")
```

## CLI Usage

### Create Evaluation

```bash
ads aqua evaluation create \
  --evaluation_source_id "ocid1.datasciencemodeldeployment.oc1.iad.xxx" \
  --evaluation_name "llama-eval-bertscore" \
  --dataset_path "oci://my-bucket@my-namespace/datasets/eval_data.jsonl" \
  --report_path "oci://my-bucket@my-namespace/eval-reports/" \
  --model_parameters '{"max_tokens": 500, "temperature": 0.7}' \
  --shape_name "VM.Standard.E4.Flex" \
  --block_storage_size 50 \
  --compartment_id "ocid1.compartment.oc1..xxx" \
  --project_id "ocid1.datascienceproject.oc1.iad.xxx" \
  --metrics '[{"name": "bertscore"}, {"name": "rouge"}]'
```

### List / Get Evaluations

```bash
ads aqua evaluation list --compartment_id "ocid1.compartment.oc1..xxx"
ads aqua evaluation get --eval_id "ocid1.datasciencemodel.oc1.iad.xxx"
```

## Interpreting Results

### BERTScore

The evaluation produces a Model Catalog entry with:
- **Precision**: How much of the generated text is semantically represented in the reference
- **Recall**: How much of the reference text is captured by the generated text
- **F1**: Harmonic mean of precision and recall

Higher scores = better quality. Scores clustered around the mean indicate consistent performance.

### ROUGE

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

### BERTScore Limitations

- May favor models mirroring its own architecture
- Lacks consideration for sentence-level syntax
- Diminished effectiveness for context beyond word-level (idioms, cultural nuances)
- Not suitable for evaluating coding models on programming tasks

## Key Source Files

- `ads/aqua/evaluation/evaluation.py` — `AquaEvaluationApp` (create, list, get, load_metrics)
- `ads/aqua/evaluation/entities.py` — `CreateAquaEvaluationDetails`, `AquaEvalMetrics`
- `ads/aqua/config/evaluation/evaluation_service_config.py` — Metric configuration
