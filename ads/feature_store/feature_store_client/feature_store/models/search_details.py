# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class SearchDetails(object):
    """
    Search details
    """

    #: A constant which can be used with the search_type property of a SearchDetails.
    #: This constant has a value of "FREE_TEXT"
    SEARCH_TYPE_FREE_TEXT = "FREE_TEXT"

    def __init__(self, **kwargs):
        """
        Initializes a new SearchDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param search_type:
            The value to assign to the search_type property of this SearchDetails.
            Allowed values for this property are: "FREE_TEXT"
        :type search_type: str

        :param text:
            The value to assign to the text property of this SearchDetails.
        :type text: str

        """
        self.swagger_types = {
            'search_type': 'str',
            'text': 'str'
        }

        self.attribute_map = {
            'search_type': 'searchType',
            'text': 'text'
        }

        self._search_type = None
        self._text = None

    @property
    def search_type(self):
        """
        **[Required]** Gets the search_type of this SearchDetails.
        Type of search you want to perform

        Allowed values for this property are: "FREE_TEXT"


        :return: The search_type of this SearchDetails.
        :rtype: str
        """
        return self._search_type

    @search_type.setter
    def search_type(self, search_type):
        """
        Sets the search_type of this SearchDetails.
        Type of search you want to perform


        :param search_type: The search_type of this SearchDetails.
        :type: str
        """
        allowed_values = ["FREE_TEXT"]
        if not value_allowed_none_or_none_sentinel(search_type, allowed_values):
            raise ValueError(
                "Invalid value for `search_type`, must be None or one of {0}"
                .format(allowed_values)
            )
        self._search_type = search_type

    @property
    def text(self):
        """
        Gets the text of this SearchDetails.
        The text to search for


        :return: The text of this SearchDetails.
        :rtype: str
        """
        return self._text

    @text.setter
    def text(self, text):
        """
        Sets the text of this SearchDetails.
        The text to search for


        :param text: The text of this SearchDetails.
        :type: str
        """
        self._text = text

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
