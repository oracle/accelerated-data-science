#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ads.data_labeling.boundingbox import BoundingBoxItem
from ads.data_labeling.ner import NERItem
from ads.data_labeling.reader.dataset_reader import LabeledDatasetReader
from ads.data_labeling.visualizer.image_visualizer import _df_to_bbox_items
from ads.data_labeling.visualizer.text_visualizer import _df_to_ner_items
from PIL import Image


class TestDataLabelingAccessMixin:
    """Test DataLabelingAccessMixin"""

    @patch.object(LabeledDatasetReader, "from_export")
    @patch("ads.common.auth.default_signer")
    def test_read_labeled_data_from_export(self, mock_signer, mock_from_export):
        mock_from_export.return_value = MagicMock()
        mock_from_export.return_value.read = MagicMock(return_value=pd.DataFrame())
        df = pd.DataFrame.ads.read_labeled_data(path="local_path.jsonl")
        mock_from_export.assert_called_once_with(
            encoding="utf-8",
            materialize=False,
            path="local_path.jsonl",
            auth=mock_signer(),
            include_unlabeled=False,
        )
        mock_from_export.return_value.read.assert_called_once()

    @patch.object(LabeledDatasetReader, "from_DLS")
    @patch("ads.common.auth.default_signer")
    def test_read_labeled_data_from_dls(self, mock_signer, mock_from_dls):
        mock_from_dls.return_value = MagicMock()
        mock_from_dls.return_value.read = MagicMock(return_value=pd.DataFrame())
        df = pd.DataFrame.ads.read_labeled_data(
            dataset_id="ocid.1234",
            compartment_id="ocid.5678",
        )
        mock_from_dls.assert_called_once_with(
            encoding="utf-8",
            materialize=False,
            dataset_id="ocid.1234",
            compartment_id="ocid.5678",
            auth=mock_signer(),
            include_unlabeled=False,
        )
        mock_from_dls.return_value.read.assert_called_once()

    @pytest.mark.skipif(
        "NoDependency" in os.environ, reason="skip for dependency test: Ipython"
    )
    def test_render_ner(self):
        """Tests rendering NER dataset to Html format."""

        test_record = [
            "Houston area contractor seeking a Sheet Metal Superintendent to supervise a crew of sheet metal mechanics.",
            [
                NERItem(label="yes", offset=0, length=12),
                NERItem(label="yes", offset=13, length=10),
            ],
        ]
        test_df = pd.DataFrame([test_record], columns=["Content", "Annotations"])
        test_ner_items = _df_to_ner_items(
            df=test_df, content_column="Content", annotations_column="Annotations"
        )

        with patch(
            "ads.data_labeling.visualizer.text_visualizer._df_to_ner_items"
        ) as mock__df_to_ner_items:
            mock__df_to_ner_items.return_value = test_ner_items
            with patch(
                "ads.data_labeling.visualizer.text_visualizer.render"
            ) as mock_render:
                mock_render.return_value = ""
                test_df.ads.render_ner(
                    content_column="Content", annotations_column="Annotations"
                )
                mock__df_to_ner_items.assert_called_once()
                mock_render.assert_called_with(items=test_ner_items, options=None)

    def test_render_bounding_box(self):
        """Tests rendering bounding box dataset."""

        test_record = [
            Image.open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "./data_label_test_files/fish.jpg",
                )
            ),
            [
                BoundingBoxItem(
                    top_left=(0.3, 0.4),
                    bottom_left=(0.3, 0.09),
                    bottom_right=(0.86, 0.09),
                    top_right=(0.86, 0.4),
                    labels=["dolphin"],
                )
            ],
        ]

        test_df = pd.DataFrame([test_record], columns=["Content", "Annotations"])

        test_bbox_items = _df_to_bbox_items(
            df=test_df, content_column="Content", annotations_column="Annotations"
        )

        with patch(
            "ads.data_labeling.visualizer.image_visualizer._df_to_bbox_items"
        ) as mock__df_to_bbox_items:
            mock__df_to_bbox_items.return_value = test_bbox_items
            with patch(
                "ads.data_labeling.visualizer.image_visualizer.render"
            ) as mock_render:
                test_df.ads.render_bounding_box(
                    content_column="Content", annotations_column="Annotations"
                )
                mock__df_to_bbox_items.assert_called_once()
                mock_render.assert_called_with(test_bbox_items, options=None, path=None)
