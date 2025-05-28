#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import typing
from dataclasses import dataclass, fields
from typing import Dict

from oci.data_science.models import (
    ContainerSummary,
    ModelDeployWorkloadConfigurationDetails,
    JobRunWorkloadConfigurationDetails,
    JobRunUseCaseConfigurationDetails,
    WorkloadConfigurationDetails,
    GenericJobRunUseCaseConfigurationDetails,
)


@dataclass(repr=False)
class MockData:
    """Used for testing serializing dataclass in handler."""

    id: str = ""
    name: str = ""


@dataclass(repr=False)
class ServiceManagedContainers:
    use_case_configuration = GenericJobRunUseCaseConfigurationDetails(
        additional_configurations={
            "metrics": '[{"task":["text-generation"],"key":"bertscore","name":"BERTScore","description":"BERTScoreisametricforevaluatingthequalityoftextgenerationmodels,suchasmachinetranslationorsummarization.Itutilizespre-trainedBERTcontextualembeddingsforboththegeneratedandreferencetexts,andthencalculatesthecosinesimilaritybetweentheseembeddings.","args":{},"tags":[]},{"task":["text-generation"],"key":"rouge","name":"ROUGEScore","description":"ROUGEscorescompareacandidatedocumenttoacollectionofreferencedocumentstoevaluatethesimilaritybetweenthem.Themetricsrangefrom0to1,withhigherscoresindicatinggreatersimilarity.ROUGEismoresuitableformodelsthatdon\'tincludeparaphrasinganddonotgeneratenewtextunitsthatdon\'tappearinthereferences.","args":{},"tags":[]},{"task":["text-generation"],"key":"bleu","name":"BLEUScore","description":"BLEU(BilingualEvaluationUnderstudy)isanalgorithmforevaluatingthequalityoftextwhichhasbeenmachine-translatedfromonenaturallanguagetoanother.Qualityisconsideredtobethecorrespondencebetweenamachine\'soutputandthatofahuman:\'thecloseramachinetranslationistoaprofessionalhumantranslation,thebetteritis\'.","args":{},"tags":[]},{"task":["text-generation"],"key":"perplexity_score","name":"PerplexityScore","description":"Perplexityisametrictoevaluatethequalityoflanguagemodels,particularlyfor\\"TextGeneration\\"tasktype.PerplexityquantifieshowwellaLLMcanpredictthenextwordinasequenceofwords.AhighperplexityscoreindicatesthattheLLMisnotconfidentinitstextgeneration—thatis,themodelis\\"perplexed\\"—whereasalowperplexityscoreindicatesthattheLLMisconfidentinitsgeneration.","args":{},"tags":[]},{"task":["text-generation"],"key":"text_readability","name":"TextReadability","description":"Textquality/readabilitymetricsoffervaluableinsightsintothequalityandsuitabilityofgeneratedresponses.MonitoringthesemetricshelpsensurethatLanguageModel(LLM)outputsareclear,concise,andappropriateforthetargetaudience.Evaluatingtextcomplexityandgradelevelhelpstailorthegeneratedcontenttotheintendedreaders.Byconsideringaspectssuchassentencestructure,vocabulary,anddomain-specificneeds,wecanmakesuretheLLMproducesresponsesthatmatchthedesiredreadinglevelandprofessionalcontext.Additionally,metricslikesyllablecount,wordcount,andcharactercountallowyoutokeeptrackofthelengthandstructureofthegeneratedtext.","args":{},"tags":[]}]',
            "shapes": '[{"name":"VM.Standard.E3.Flex","ocpu":8,"memory_in_gbs":128,"block_storage_size":200,"filter":{"evaluation_container":["odsc-llm-evaluate"],"evaluation_target":["datasciencemodeldeployment"]}},{"name":"VM.Standard.E4.Flex","ocpu":8,"memory_in_gbs":128,"block_storage_size":200,"filter":{"evaluation_container":["odsc-llm-evaluate"],"evaluation_target":["datasciencemodeldeployment"]}},{"name":"VM.Standard3.Flex","ocpu":8,"memory_in_gbs":128,"block_storage_size":200,"filter":{"evaluation_container":["odsc-llm-evaluate"],"evaluation_target":["datasciencemodeldeployment"]}},{"name":"VM.Optimized3.Flex","ocpu":8,"memory_in_gbs":128,"block_storage_size":200,"filter":{"evaluation_container":["odsc-llm-evaluate"],"evaluation_target":["datasciencemodeldeployment"]}}]',
        }
    )

    workload_configuration = JobRunWorkloadConfigurationDetails(
        use_case_configuration=use_case_configuration
    )
    MOCK_OUTPUT = [
        ContainerSummary(
            **{
                "container_name": "odsc-vllm-serving",
                "display_name": "VLLM:0.6.4.post1.2",
                "family_name": "odsc-vllm-serving",
                "description": "This container is used for llm inference, batch inference and serving",
                "is_latest": True,
                "target_workloads": ["MODEL_DEPLOYMENT", "JOB_RUN"],
                "usages": ["INFERENCE", "BATCH_INFERENCE", "OTHER"],
                "tag": "0.6.4.post1.2",
                "lifecycle_state": "ACTIVE",
                "workload_configuration_details_list": [
                    ModelDeployWorkloadConfigurationDetails(
                        **{
                            "cmd": "--served-model-name odsc-llm --disable-custom-all-reduce --seed 42 ",
                            "server_port": 8080,
                            "health_check_port": 8080,
                            "additional_configurations": {
                                "HEALTH_CHECK_PORT": "8080",
                                "MODEL_DEPLOY_PREDICT_ENDPOINT": "/v1/completions",
                                "PORT": "8080",
                                "modelFormats": "SAFETENSORS",
                                "platforms": "NVIDIA_GPU",
                                "restrictedParams": '["--port","--host","--served-model-name","--seed"]',
                            },
                        }
                    )
                ],
                "tag_configuration_list": [],
                "freeform_tags": None,
                "defined_tags": None,
            }
        ),
        ContainerSummary(
            **{
                "container_name": "odsc-llm-fine-tuning",
                "display_name": "Fine-Tune:2.2.62.70",
                "family_name": "odsc-llm-fine-tuning",
                "description": "This container is used to fine tune llm",
                "is_latest": True,
                "target_workloads": ["JOB_RUN"],
                "usages": ["FINE_TUNE"],
                "tag": "2.2.62.70",
                "lifecycle_state": "ACTIVE",
                "workload_configuration_details_list": [],
                "tag_configuration_list": [],
                "freeform_tags": None,
                "defined_tags": None,
            }
        ),
        ContainerSummary(
            **{
                "container_name": "odsc-llm-evaluate",
                "display_name": "Evaluate:0.1.3.4",
                "family_name": "odsc-llm-evaluate",
                "description": "This container supports evaluation on model deployment",
                "is_latest": True,
                "target_workloads": ["JOB_RUN"],
                "usages": ["EVALUATION"],
                "tag": "0.1.3.4",
                "lifecycle_state": "ACTIVE",
                "workload_configuration_details_list": [workload_configuration],
                "tag_configuration_list": [],
                "freeform_tags": None,
                "defined_tags": None,
            }
        ),
    ]


class HandlerTestDataset:
    MOCK_OCID = "ocid.datasciencemdoel.<ocid>"
    mock_valid_input = dict(
        evaluation_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
        evaluation_name="test_evaluation_name",
        dataset_path="oci://dataset_bucket@namespace/prefix/dataset.jsonl",
        report_path="oci://report_bucket@namespace/prefix/",
        model_parameters=dict(max_token=500),
        shape_name="VM.Standard.E3.Flex",
        block_storage_size=1,
        experiment_name="test_experiment_name",
        memory_in_gbs=1,
        ocpus=1.0,
    )
    mock_invalid_input = dict(name="myvalue")
    mock_dataclass_obj = MockData(id="myid", name="myname")
    mock_service_payload_create = {
        "target_service": "data_science",
        "status": 404,
        "code": "NotAuthenticated",
        "opc-request-id": "1234",
        "message": "The required information to complete authentication was not provided or was incorrect.",
        "operation_name": "create_resources",
        "timestamp": "2024-04-12T02:51:24.977404+00:00",
        "request_endpoint": "POST xxx",
    }
    mock_service_payload_get = {
        "target_service": "data_science",
        "status": 404,
        "code": "NotAuthenticated",
        "opc-request-id": "1234",
        "message": "The required information to complete authentication was not provided or was incorrect.",
        "operation_name": "get_job_run",
        "timestamp": "2024-04-12T02:51:24.977404+00:00",
        "request_endpoint": "GET xxx",
    }

    def mock_url(self, action):
        return f"{self.MOCK_OCID}/{action}"


@dataclass
class BaseFormat:
    """Implements type checking for each format."""

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = (
                field.type.__origin__
                if isinstance(field.type, typing._GenericAlias)
                else field.type
            )
            if not isinstance(value, field_type):
                raise TypeError(
                    f"Expected {field.name} to be {field_type}, " f"got {repr(value)}"
                )


def check(conf_schema, conf):
    """Check if the format of the output dictionary is correct."""
    try:
        conf_schema(**conf)
        return True
    except TypeError:
        return False
