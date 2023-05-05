#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from datetime import datetime, timezone, timedelta
from ads.catalog.summary import SummaryList
import unittest
import ads.common.utils as utils


class SummaryListTest(unittest.TestCase):
    """Contains test cases for ads.catalog.model.py"""

    rand_datetime = datetime(2020, 6, 1, 18, 24, 42, 110000, tzinfo=timezone.utc)
    different_date_format = "%d-%m-%Y %H:%M:%S"

    @staticmethod
    def generate_entity_list(
        list_len=1,
        date_time=datetime(2020, 7, 1, 18, 24, 42, 110000, tzinfo=timezone.utc),
        collision_value="",
        drop_response_value=None,
    ):

        random_list = []

        for i in range(list_len):
            formatted_datetime = date_time.isoformat()
            entity_item = {
                "compartment_id": utils.random_valid_ocid(
                    prefix="ocid1.compartment.oc1.<unique_ocid>"
                ),
                "description": {},
                "display_name": "sample new df app",
                "freeform_tags": {},
                "project_id": "".join(
                    [
                        utils.random_valid_ocid(
                            prefix="ocid1.dataflowapplication.oc1.iad.<unique_ocid>"
                        ),
                        collision_value,
                    ]
                ),
                "lifecycle_state": "ACTIVE",
                "user_name": "sample.user@example.com",
                "time_created": formatted_datetime,
                "created_by": "",
                "id": "".join(
                    [
                        utils.random_valid_ocid(
                            prefix="ocid1.catalog.oc1.iad.<unique_ocid>"
                        ),
                        collision_value,
                    ]
                ),
            }

            if drop_response_value is not None:
                del entity_item[drop_response_value]

            random_list.append(entity_item)
            date_time += timedelta(minutes=1)

        return random_list

    def test_empty_summary_list(self):
        """Test initialize summary list with empty list."""
        empty_list = []
        sl = SummaryList(empty_list)
        assert sl.df.empty

    def test_summary_list_to_dataframe_with_different_datetime_format(self):
        """Test initialize summary list with different datetime format"""
        # test SummaryList with different datetime format
        date_time = self.rand_datetime
        test_list = self.generate_entity_list(list_len=1, date_time=date_time)
        sl_diff_date_format = SummaryList(
            test_list, datetime_format=self.different_date_format
        )

        assert sl_diff_date_format.datetime_format == self.different_date_format

        # test to_SummaryList().dataframe() method with different datetime format
        sl = SummaryList(test_list)
        df_changed_date_format = sl.to_dataframe(
            datetime_format=self.different_date_format
        )
        expected_date = date_time.strftime(self.different_date_format)

        assert df_changed_date_format["time_created"].values[0] == expected_date

    def test_summary_list_collision_protection(self):
        """Test summary list collision protection"""
        collision_value = "abcdef"
        test_list = self.generate_entity_list(
            list_len=10, collision_value=collision_value
        )
        sl = SummaryList(test_list, datetime_format=utils.date_format)

        # check length of shortened index value is bigger then length of non unique values at the end of "id"
        id_len = len(sl.df.index[0])
        assert id_len > len(collision_value)

    def test_ordered_column_not_in_summary_list(self):
        """Test summary list initialize with list which does not have attributes in ordered_columns."""
        # here we drop one of ordered_columns = ['display_name', 'time_created', 'lifecycle_state']
        test_list_with_dropped_value = self.generate_entity_list(
            list_len=2, drop_response_value="display_name"
        )
        sl = SummaryList(
            test_list_with_dropped_value, datetime_format=utils.date_format
        )

        assert "display_name" not in sl.df.columns

    def test_color_lifecycle_state(self):
        """Test SummaryList._color_lifecycle_state method."""
        test_list = self.generate_entity_list(list_len=1)
        sl = SummaryList(test_list, datetime_format=utils.date_format)
        assert "color: %s" % "green" == sl._color_lifecycle_state("ACTIVE")
        assert "color: %s" % "grey" == sl._color_lifecycle_state("INACTIVE")
        assert "color: %s" % "black" == sl._color_lifecycle_state("DELETING")
        assert "color: %s" % "blue" == sl._color_lifecycle_state("CREATING")
        assert "color: %s" % "black" == sl._color_lifecycle_state("DELETED")
        assert "color: %s" % "red" == sl._color_lifecycle_state("FAILED")
