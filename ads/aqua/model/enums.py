#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from ads.common.extended_enum import ExtendedEnum


class FineTuningDefinedMetadata(ExtendedEnum):
    """Represents the defined metadata keys used in Fine Tuning."""

    VAL_SET_SIZE = "val_set_size"
    TRAINING_DATA = "training_data"


class FineTuningCustomMetadata(ExtendedEnum):
    """Represents the custom metadata keys used in Fine Tuning."""

    FT_SOURCE = "fine_tune_source"
    FT_SOURCE_NAME = "fine_tune_source_name"
    FT_OUTPUT_PATH = "fine_tune_output_path"
    FT_JOB_ID = "fine_tune_job_id"
    FT_JOB_RUN_ID = "fine_tune_jobrun_id"
    TRAINING_METRICS_FINAL = "train_metrics_final"
    VALIDATION_METRICS_FINAL = "val_metrics_final"
    TRAINING_METRICS_EPOCH = "train_metrics_epoch"
    VALIDATION_METRICS_EPOCH = "val_metrics_epoch"


class MultiModelSupportedTaskType(ExtendedEnum):
    TEXT_GENERATION = "text_generation"
    TEXT_GENERATION_INFERENCE = "text_generation_inference"
    TEXT2TEXT_GENERATION = "text2text_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATIONAL = "conversational"
    FEATURE_EXTRACTION = "feature_extraction"
    SENTENCE_SIMILARITY = "sentence_similarity"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic_speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_TEXT_TO_TEXT = "image_text_to_text"
    CODE_SYNTHESIS = "code_synthesis"
    QUESTION_ANSWERING = "question_answering"
    AUDIO_CLASSIFICATION = "audio_classification"
    AUDIO_TO_AUDIO = "audio_to_audio"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_TO_TEXT = "image_to_text"
    IMAGE_TO_IMAGE = "image_to_image"
    VIDEO_CLASSIFICATION = "video_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
