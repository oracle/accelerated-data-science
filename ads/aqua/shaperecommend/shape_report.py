#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Optional

from pydantic import BaseModel, Field


class RequestRecommend(BaseModel):
    model: str
    max_model_len: Optional[int] = Field(4096, )

class GPUSummary(BaseModel):
    gpu_count: int
    gpu_memory_in_gb: int
    limiting_factor: str


class ShapeSummary(BaseModel):
    shape: str
    gpu_reports: List[GPUSummary]


class DeploymentShapeSummary(BaseModel):
    batch_size: int
    max_seq_len: int
    precision: str
    gb_used_by_model: float
    shape_reports: List[ShapeSummary]


class TroubleshootShapeSummary(BaseModel):
    largest_shape: str
    gpu_memory_in_gb: int
    gb_used_by_model: float
    batch_size: int
    max_seq_len: int
    precision: str
    advice: str


class ShapeRecommendationReport(BaseModel):
    """
    Contains shape fit recommendations and an optional troubleshooting summary.
    """

    # Each entry is: for this batch_size and max_seq_len, here are valid shapes
    recommendations: List[DeploymentShapeSummary] = []
    troubleshoot: Optional[TroubleshootShapeSummary] = None

