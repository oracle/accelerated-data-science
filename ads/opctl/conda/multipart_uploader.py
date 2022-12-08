#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List

from ads.common.oci_client import OCIClientFactory
from ads.common.auth import create_signer
from ads.opctl.utils import parse_conda_uri
import oci
import mmap
import json
import time
from tqdm import tqdm
from threading import Thread
import os


class MultiPartUploader:
    """
    Class that implements multipart uploading for conda packs.
    """

    def __init__(
        self,
        source_file: str,
        dst_uri: str,
        parts: int,
        oci_config: str = None,
        oci_profile: str = None,
        auth_type: str = None,
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        source_file : str
            path to conda pack file
        dst_uri : str
            path to destination of object storage location
        parts : int
            number of parts
        oci_config : str, optional
            path to oci config file, by default None
        oci_profile : str, optional
            oci profile to use, by default None
        auth_type : str
            authentication method, by default None
        """
        self.src = source_file
        self.dst = dst_uri
        self.oci_auth = create_signer(auth_type, oci_config, oci_profile)
        self.client = OCIClientFactory(**self.oci_auth).object_storage
        self.ns, self.bucket, self.path, _ = parse_conda_uri(dst_uri)
        self.file_size = os.path.getsize(self.src)
        # mmap offset arg must be a multiple of the ALLOCATIONGRANULARITY, change chunk_size to control offset value
        self.chunk_size = (
            (self.file_size // parts)
            // mmap.ALLOCATIONGRANULARITY
            * mmap.ALLOCATIONGRANULARITY
        )

    def upload(self, opc_meta: Dict = None) -> bool:
        """Uploading a conda pack to object storage.

        Parameters
        ----------
        opc_meta : Dict, optional
            metadata dictionary, by default None

        Returns
        -------
        bool
            whether uploading was successful

        Raises
        ------
        RuntimeError
            Uploading failed
        """
        multipart_upload_details = (
            oci.object_storage.models.CreateMultipartUploadDetails()
        )
        multipart_upload_details.object = self.path
        multipart_upload_details.metadata = {"opc-meta-manifest": json.dumps(opc_meta)}

        upload_details = self.client.create_multipart_upload(
            self.ns, self.bucket, multipart_upload_details
        )
        uploaded_parts = []
        threads = []
        mm_objects = []
        responses = []
        upload_id = upload_details.data.upload_id
        print(f"The upload id is {upload_id}.")
        with open(self.src, "rb") as pkf:
            counter = 0
            for offset, length in self._chunks():
                counter += 1
                mm = mmap.mmap(
                    pkf.fileno(), length, access=mmap.ACCESS_READ, offset=offset
                )
                mm_objects.append((mm, length))
                t = Thread(
                    target=self._upload_chunk,
                    args=(upload_id, counter, mm, uploaded_parts, responses),
                )
                t.start()
                threads.append(t)
            t = Thread(
                target=self._track_progress, args=(mm_objects, counter, responses)
            )
            t.start()
            threads.append(t)
            for t in threads:
                t.join()
        # Successful responses HTTP status in range (200–299)
        successful = all(200 <= r.status < 300 for r in responses)
        if successful:
            self.client.commit_multipart_upload(
                self.ns,
                self.bucket,
                self.path,
                upload_details.data.upload_id,
                oci.object_storage.models.CommitMultipartUploadDetails(
                    parts_to_commit=uploaded_parts
                ),
            )
            print(f"{self.src} uploaded successfuly to {self.dst}.")
        else:
            for r in responses:
                print(r.status, r.headers)
            raise RuntimeError(f"{self.src} upload failed.")
        return successful

    def _chunks(self):
        start_position = 0
        while start_position < self.file_size:
            yield start_position, min(self.chunk_size, self.file_size - start_position)
            start_position += self.chunk_size

    def _upload_chunk(
        self,
        upload_id: str,
        counter: int,
        mm: mmap.mmap,
        uploaded_parts: List,
        responses: List,
    ) -> None:
        response = self.client.upload_part(
            self.ns, self.bucket, self.path, upload_id, counter, mm
        )
        responses.append(response)
        uploaded_parts.append(
            oci.object_storage.models.CommitMultipartUploadPartDetails(
                etag=response.headers["etag"], part_num=counter
            )
        )

    @staticmethod
    def _track_progress(
        mm_objects: List[mmap.mmap], counter: int, responses: List
    ) -> None:
        nresponses = 0
        progress_objs = []
        for index, (mm_object, length) in enumerate(mm_objects):
            tqdm_obj = tqdm(
                total=length,
                unit="B",
                unit_scale=True,
                unit_divisor=2**10,
                desc=f"Part {index + 1}",
                position=index + 1,
                leave=True,
            )
            progress_objs.append([mm_object, tqdm_obj, 0, length])
        while nresponses < counter:
            for obj in progress_objs:
                current_pos = obj[0].tell()
                obj[1].update(current_pos - obj[2])
                obj[1].refresh()
                obj[2] = current_pos
            time.sleep(2)
            nresponses = len(responses)
        map(lambda x: x[1].refresh(), progress_objs)
        map(lambda x: x[1].close(), progress_objs)
