import json
import ads.common
import oci
import os
import ads
from ads.model.datascience_model import DataScienceModel
from typing import List, Optional
import logging

logger = logging.getLogger("ads.model_description")
logger.setLevel(logging.INFO)


class DataScienceModelCollection(DataScienceModel):
    
    def _auth(self):
        """
        Internal method that authenticates the model description instance by initializing OCI clients.

        Parameters:
        - None

        Returns:
        - None

        Note:
        - This method retrieves authentication data using default signer from the `ads.common.auth` module.
        - The region information is extracted from the authentication data.
        """
        authData = ads.common.auth.default_signer()
        signer = authData["signer"]
        self.region = authData["config"]["region"]

        # data science client
        self.data_science_client = oci.data_science.DataScienceClient(
            {"region": self.region}, signer=signer
        )
        # oss client
        self.object_storage_client = oci.object_storage.ObjectStorageClient(
            {"region": self.region}, signer=signer
        )

    def __init__(self, spec: ads.Dict = None, **kwargs) -> None:
        super().__init__(spec, **kwargs)
     
        self.empty_json = {
            "version": "1.0",
            "type": "modelOSSReferenceDescription",
            "models": [],
        }
        self.region = ""
        self._auth()

        self.set_spec(self.CONST_MODEL_FILE_DESCRIPTION, self.empty_json)

    def with_ref_model_id(self, model_ocid: str):

        # if model given then get that as the starting reference point
        logger.info("Getting model details from backend")
        try:
            get_model_artifact_content_response = (
                self.data_science_client.get_model_artifact_content(
                    model_id=model_ocid,
                )
            )
            content = get_model_artifact_content_response.data.content
            self.set_spec(self.CONST_MODEL_FILE_DESCRIPTION, json.loads(content))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e
        return self
    
    def add(self, namespace: str, bucket: str, prefix: Optional[str] =None, files: Optional[List[str]] =None):
        """
        Adds information about objects in a specified bucket to the model description JSON.

        Parameters:
        - namespace (str): The namespace of the object storage.
        - bucket (str): The name of the bucket containing the objects.
        - prefix (str, optional): The prefix used to filter objects within the bucket. Defaults to None.
        - files (list of str, optional): A list of file names to include in the model description.
        If provided, only objects with matching file names will be included. Defaults to None.

        Returns:
        - None

        Raises:
        - ValueError: If no files are found to add to the model description.

        Note:
        - If `files` is not provided, it retrieves information about all objects in the bucket.
        If `files` is provided, it only retrieves information about objects with matching file names.
        - If no objects are found to add to the model description, a ValueError is raised.
        """

        # Remove if the model already exists
        self.remove(namespace, bucket, prefix)

        def check_if_file_exists(fileName):
            isExists = False
            try:
                headResponse = self.object_storage_client.head_object(
                    namespace, bucket, object_name=fileName
                )
                if headResponse.status == 200:
                    isExists = True
            except Exception as e:
                if hasattr(e, "status") and e.status == 404:
                    logger.error(f"File not found in bucket: {fileName}")
                else:
                    logger.error(f"An error occured: {e}")
            return isExists

        # Function to un-paginate the api call with while loop
        def list_obj_versions_unpaginated():
            objectStorageList = []
            has_next_page, opc_next_page = True, None
            while has_next_page:
                response = self.object_storage_client.list_object_versions(
                    namespace_name=namespace,
                    bucket_name=bucket,
                    prefix=prefix,
                    fields="name,size",
                    page=opc_next_page,
                )
                objectStorageList.extend(response.data.items)
                has_next_page = response.has_next_page
                opc_next_page = response.next_page
            return objectStorageList

        # Fetch object details and put it into the objects variable
        objectStorageList = []
        if files == None:
            objectStorageList = list_obj_versions_unpaginated()
        else:
            for fileName in files:
                if check_if_file_exists(fileName=fileName):
                    objectStorageList.append(
                        self.object_storage_client.list_object_versions(
                            namespace_name=namespace,
                            bucket_name=bucket,
                            prefix=fileName,
                            fields="name,size",
                        ).data.items[0]
                    )

        objects = [
            {"name": obj.name, "version": obj.version_id, "sizeInBytes": obj.size}
            for obj in objectStorageList
            if obj.size > 0
        ]

        if len(objects) == 0:
            error_message = (
                f"No files to add in the bucket: {bucket} with namespace: {namespace} "
                f"and prefix: {prefix}. File names: {files}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        tmp_model_file_description = self.model_file_description
        tmp_model_file_description['models'].append({
                "namespace": namespace,
                "bucketName": bucket,
                "prefix": prefix,
                "objects": objects,
            })
        self.set_spec(self.CONST_MODEL_FILE_DESCRIPTION, tmp_model_file_description)
    
    def remove(self, namespace: str, bucket: str, prefix: Optional[str]=None):
        """
        Removes information about objects in a specified bucket from the model description JSON.

        Parameters:
        - namespace (str): The namespace of the object storage.
        - bucket (str): The name of the bucket containing the objects.
        - prefix (str, optional): The prefix used to filter objects within the bucket. Defaults to None.

        Returns:
        - None

        Note:
        - This method removes information about objects in the specified bucket from the
        instance of the ModelDescription.
        - If a matching model (with the specified namespace, bucket name, and prefix) is found
        in the model description JSON, it is removed.
        - If no matching model is found, the method returns without making any changes.
        """

        def findModelIdx():
            for idx, model in enumerate(self.model_file_description["models"]):
                if (
                    model["namespace"],
                    model["bucketName"],
                    (model["prefix"] if ("prefix" in model) else None),
                ) == (namespace, bucket, prefix):
                    return idx
            return -1

        modelSearchIdx = findModelIdx()
        if modelSearchIdx == -1:
            return
        else:
            # model found case
            self.model_file_description["models"].pop(modelSearchIdx)

    def create(self):
        """
        Saves the model to the Model Catalog of Oracle Cloud Infrastructure (OCI) Data Science service.

        Parameters:
        - project_ocid (str): The OCID (Oracle Cloud Identifier) of the OCI Data Science project.
        - compartment_ocid (str): The OCID of the compartment in which the model will be created.
        - display_name (str, optional): The display name for the created model. If not provided,
        a default display name indicating the creation timestamp is used. Defaults to None.

        Returns:
        - str: The OCID of the created model.

        Note:
        - The display name defaults to a string indicating the creation timestamp if not provided.
        """
        tmp_file_path = self.build()
        self = self.with_artifact(uri=tmp_file_path)
        created_model = super().create()
        try:
            os.remove(tmp_file_path)
        except Exception as e:
            logger.error(f"Error occurred while cleaning file: {e}")
            raise e
        return created_model.id

    def build(self) -> str:
        """
        Builds the model description JSON and writes it to a file.

        Parameters:
        - None

        Returns:
        - str: The absolute file path where the model description JSON is stored.

        Note:
        - This method serializes the current model description attribute to a JSON file named 'resultModelDescription.json' with an indentation of 2 spaces.
        """
        logger.info("Building...")
        file_path = "resultModelDescription.json"
        try:
            with open(file_path, "w") as json_file:
                json.dump(self.model_file_description, json_file, indent=2)
        except IOError as e:
            logger.error(
                f"Error writing to file '{file_path}': {e}"
            )  # Handle the exception accordingly, e.g., log the error, retry writing, etc.
        except Exception as e:
            logger.error(
                f"An unexpected error occurred: {e}"
            )  # Handle other unexpected exceptions
        logger.info("Model Artifact stored successfully.")
        return os.path.abspath(file_path)

    def show(self):
        """
        Displays the current model description JSON in a human-readable format.

        Parameters:
        - None

        Returns:
        - str: The json representation of current model artifact

        Note:
        - The JSON representation of the model description is formatted with an indentation
        of 4 spaces.
        """
        logger.info(json.dumps(self.model_file_description, indent=4))
        return json.dumps(self.model_file_description, indent=4)