#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.feature_store.common.utils.utility import (
    largest_matching_subset_of_primary_keys,
)


class QueryValidator:
    @staticmethod
    def validate_query_join(left_feature_group, join):
        left_features = [f.feature_name for f in left_feature_group.features]
        right_feature_group = join.sub_query.left_feature_group
        right_features = [f.feature_name for f in right_feature_group.features]

        if join.on:
            # Validate keys in the explicit "ON" clause
            for key in join.on:
                if key not in left_features:
                    raise ValueError(
                        f"Cannot join feature group '{left_feature_group.name}' on '{key}', as it is not present in "
                        f"the feature group. "
                    )
                if key not in right_features:
                    raise ValueError(
                        f"Cannot join feature group '{right_feature_group.name}' on '{key}', as it is not present in "
                        f"the feature group. "
                    )

        elif join.left_on and join.right_on:
            # Validate keys in the "LEFT ON" and "RIGHT ON" clauses
            for key in join.left_on:
                if key not in left_features:
                    raise ValueError(
                        f"Cannot join feature group '{left_feature_group.name}' on '{key}', as it is not present in "
                        f"the feature group. "
                    )
            for key in join.right_on:
                if key not in right_features:
                    raise ValueError(
                        f"Cannot join feature group '{right_feature_group.name}' on '{key}', as it is not present in "
                        f"the feature group. "
                    )

        else:
            # Use largest_matching_subset_of_primary_keys to find matching primary keys
            matching_primary_keys = largest_matching_subset_of_primary_keys(
                left_feature_group, right_feature_group
            )
            if not matching_primary_keys:
                raise ValueError(
                    f"Cannot join feature groups '{left_feature_group.name}' and '{right_feature_group.name}', as no "
                    f"matching primary keys were found. "
                )
