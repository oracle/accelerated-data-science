#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List


class OCIRDetails:
    class InvalidOCIRUrlException(Exception):
        def __init__(self, ocir_url: str):
            self.ocir_url = ocir_url
            pass

        def __repr__(self):
            print(
                f"OCIR URL: {self.ocir_url}"
                + " is invalid. It should be of the pattern {region}.ocir.io/{tenancy_namespace}/{repository_path}"
            )

    def __init__(self, ocir_url: str):
        self._ocir_url: str = ocir_url.rstrip("/")
        ocir_arr: List[str] = ocir_url.split("/")
        if len(ocir_arr) < 3:
            raise self.InvalidOCIRUrlException(ocir_url=ocir_url)
        else:
            self._ocir_region_url = ocir_arr[0]
            self._namespace = ocir_arr[1]
            self._path_in_tenancy = "/".join(ocir_arr[2:])
            self._image = ocir_arr[-1]
            self._repository_url = "/".join(ocir_arr[:-1])

    @property
    def ocir_region_url(self) -> str:
        return self._ocir_region_url

    @property
    def path_in_tenancy(self) -> str:
        return self._path_in_tenancy

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def ocir_url(self) -> str:
        return self._ocir_url

    @property
    def image(self) -> str:
        return self._image

    @property
    def repository_url(self) -> str:
        return self._repository_url
