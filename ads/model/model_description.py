import json
import ads.common
import oci
import pytz
import datetime
import os
from oci.data_science.models import Metadata
import ads

# from ads.common import logger
import logging

logger = logging.getLogger("ads.model_description")
logger.setLevel(logging.INFO)


class ModelDescription:
    """
    Represents a model description for multi model artifacts. Provides methods to ease the process of creating such models.

    Methods:
    - add(namespace, bucket, prefix=None, files=None): Adds information about objects to the model description JSON.
    - remove(namespace, bucket, prefix=None): Removes information about objects from the model description JSON.
    - show(): Displays the current model description JSON in a human-readable format.
    - build(): Builds the model description JSON and writes it to a file.
    - save(project_ocid, compartment_ocid, display_name=None): Saves the model to the Model Catalog of OCI Data Science service.
    """

    empty_json = {
        "version": "1.0",
        "type": "modelOSSReferenceDescription",
        "models": [],
    }

    def auth(self):
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

    def __init__(self, model_ocid=None):
        """
        Initializes a new instance of the ModelDescription class.

        Parameters:
        - model_ocid (str, optional): The OCID (Oracle Cloud Identifier) of the model.
        If provided, retrieves the takes the model artifact content as starting point or else initializes from blank artifact.

        Returns:
        - None

        Raises:
        - json.JSONDecodeError: If there is an error decoding the JSON content retrieved
        from the backend.
        - Exception: If an unexpected error occurs while retrieving the model artifact content.

        Note:
        - If `model_ocid` is provided, this method attempts to retrieve the model artifact
        content from the backend using the provided OCID. If successful, it initializes
        `modelDescriptionJson` with the retrieved content. If unsuccessful, it logs the
        error and raises an exception.
        - If `model_ocid` is not provided, `modelDescriptionJson` is initialized with an
        empty JSON structure.
        """

        self.region = ""
        self.auth()

        if model_ocid == None:
            # if no model given then start from scratch
            self.modelDescriptionJson = self.empty_json
        else:
            # if model given then get that as the starting reference point
            logger.info("Getting model details from backend")
            try:
                get_model_artifact_content_response = (
                    self.data_science_client.get_model_artifact_content(
                        model_id=model_ocid,
                    )
                )
                content = get_model_artifact_content_response.data.content
                self.modelDescriptionJson = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
                raise e
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                raise e

    def add(self, namespace, bucket, prefix=None, files=None):
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

        def checkIfFileExists(fileName):
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
        def listObjectVersionsUnpaginated():
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
            objectStorageList = listObjectVersionsUnpaginated()
        else:
            for fileName in files:
                if checkIfFileExists(fileName=fileName):
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

        self.modelDescriptionJson["models"].append(
            {
                "namespace": namespace,
                "bucketName": bucket,
                "prefix": prefix,
                "objects": objects,
            }
        )

    def remove(self, namespace, bucket, prefix=None):
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
            for idx, model in enumerate(self.modelDescriptionJson["models"]):
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
            self.modelDescriptionJson["models"].pop(modelSearchIdx)

    def show(self):
        """
        Displays the current model description JSON in a human-readable format.

        Parameters:
        - None

        Returns:
        - None

        Note:
        - The JSON representation of the model description is formatted with an indentation
        of 4 spaces.
        """
        logger.info(json.dumps(self.modelDescriptionJson, indent=4))

    def build(self):
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
                json.dump(self.modelDescriptionJson, json_file, indent=2)
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

    def save(self, project_ocid, compartment_ocid, display_name=None):
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
        display_name = (
            "Created by MMS SDK on "
            + datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            if (display_name == None)
            else display_name
        )
        customMetadataList = [Metadata(key="modelDescription", value="true")]
        model_details = oci.data_science.models.CreateModelDetails(
            compartment_id=compartment_ocid,
            project_id=project_ocid,
            display_name=display_name,
            custom_metadata_list=customMetadataList,
        )
        logger.info("Created model details")
        model = self.data_science_client.create_model(model_details)
        logger.info("Created model")
        self.data_science_client.create_model_artifact(
            model.data.id,
            json.dumps(self.modelDescriptionJson),
            content_disposition='attachment; filename="modelDescription.json"',
        )
        logger.info(f"Successfully created model with OCID: {model.data.id}")
        return model.data.id
