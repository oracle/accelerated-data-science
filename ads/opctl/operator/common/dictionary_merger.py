#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
from typing import Any, Dict, List


class DictionaryMerger:
    """
    A class to update dictionary values for specified keys and
    then merge these updates back into the original dictionary.

    Example
    -------
    >>> updates = {
    ...     "infrastructure.blockStorageSize": "20",
    ...     "infrastructure.projectId": "my_new_project_id"
    ...     "runtime.conda": "my_conda"
    ... }
    >>> updater = DictionaryMerger(updates)
    >>> source_data = {
    ...     "infrastructure": {
    ...         "blockStorageSize": "10",
    ...         "projectId": "old_project_id",
    ...     },
    ...     "runtime": {
    ...         "conda": "conda",
    ...     },
    ... }
    >>> result = updater.dispatch(source_data)
    ... {
    ...     "infrastructure": {
    ...         "blockStorageSize": "20",
    ...         "projectId": "my_new_project_id",
    ...     },
    ...     "runtime": {
    ...         "conda": "my_conda",
    ...     },
    ... }

    Attributes
    ----------
    updates: Dict[str, Any]
        A dictionary containing the keys with their new values for the update.
    """

    _SYSTEM_KEYS = set(("kind", "type", "spec", "infrastructure", "runtime"))

    def __init__(self, updates: Dict[str, Any], system_keys: List[str] = None):
        """
        Initializes the DictionaryMerger with a dictionary of updates.

        Parameters
        ----------
        updates Dict[str, Any]
            A dictionary with keys that need to be updated and their new values.
        system_keys: List[str]
            The list of keys that cannot be replaced in the source dictionary.
        """
        self.updates = updates
        self.system_keys = set(system_keys or []).union(self._SYSTEM_KEYS)

    def _update_keys(
        self, dict_to_update: Dict[str, Any], parent_key: str = ""
    ) -> None:
        """
        Recursively updates the values of given keys in a dictionary.

        Parameters
        ----------
        dict_to_update: Dict[str, Any]
            The dictionary whose values are to be updated.
        parent_key: (str, optional)
            The current path in the dictionary being processed, used for nested dictionaries.

        Returns
        -------
        None
            The method updates the dict_to_update in place.
        """
        for key, value in dict_to_update.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                self._update_keys(value, new_key)
            elif new_key in self.updates and key not in self.system_keys:
                dict_to_update[key] = self.updates[new_key]

    def _merge_updates(
        self,
        original_dict: Dict[str, Any],
        updated_dict: Dict[str, Any],
        parent_key: str = "",
    ) -> None:
        """
        Merges updated values from the updated_dict into the original_dict based on the provided keys.

        Parameters
        ----------
        original_dict: Dict[str, Any]
            The original dictionary to merge updates into.
        updated_dict: Dict[str, Any]
            The updated dictionary with new values.
        parent_key: str
            The base key path for recursive merging.

        Returns
        -------
        None
            The method updates the original_dict in place.
        """
        for key, value in updated_dict.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict) and key in original_dict:
                self._merge_updates(original_dict[key], value, new_key)
            elif new_key in self.updates:
                original_dict[key] = value

    def merge(self, src_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the dictionary with new values for specified keys and merges
        these changes back into the original dictionary.

        Parameters
        ----------
        src_dict: Dict[str, Any]
            The dictionary to be updated and merged.

        Returns
        -------
        Dict[str, Any]
            The updated and merged dictionary.
        """
        if not self.updates:
            return src_dict

        original_dict = copy.deepcopy(src_dict)
        updated_dict = copy.deepcopy(src_dict)

        # Update the dictionary with the new values
        self._update_keys(updated_dict)

        # Merge the updates back into the original dictionary
        self._merge_updates(original_dict, updated_dict)

        return original_dict
