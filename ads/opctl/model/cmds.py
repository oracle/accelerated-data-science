import os
import shutil

from ads.common.auth import AuthContext
from ads.model.datascience_model import DataScienceModel
from ads.opctl import logger
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import DEFAULT_MODEL_FOLDER


def download_model(**kwargs):
    p = ConfigProcessor().step(ConfigMerger, **kwargs)
    ocid = p.config["execution"]["ocid"]

    auth_type = p.config["execution"].get("auth")
    profile = p.config["execution"].get("oci_profile", None)
    model_folder = os.path.expanduser(
        p.config["execution"].get("model_save_folder", DEFAULT_MODEL_FOLDER)
    )
    force_overwrite = p.config["execution"].get("force_overwrite", False)

    artifact_directory = os.path.join(model_folder, str(ocid))
    if (
        not os.path.exists(artifact_directory)
        or len(os.listdir(artifact_directory)) == 0
    ) or force_overwrite:
        region = p.config["execution"].get("region", None)
        bucket_uri = p.config["execution"].get("bucket_uri", None)
        timeout = p.config["execution"].get("timeout", None)
        logger.info(
            f"No cached model found. Downloading the model {ocid} to {artifact_directory}. If you already have a copy of the model, specify `artifact_directory` instead of `ocid`. You can specify `model_save_folder` to decide where to store the model artifacts."
        )
        _download_model(
            ocid=ocid,
            artifact_directory=artifact_directory,
            region=region,
            bucket_uri=bucket_uri,
            timeout=timeout,
            force_overwrite=force_overwrite,
            auth=auth_type,
            profile=profile
        )
    else:
        logger.error(f"Model already exists. Set `force_overwrite=True` to overwrite.")
        raise ValueError(
            f"Model already exists. Set `force_overwrite=True` to overwrite."
        )


def _download_model(
    ocid, artifact_directory, region, bucket_uri, timeout, force_overwrite, auth, profile=None
):
    os.makedirs(artifact_directory, exist_ok=True)
    kwargs = {"auth": auth}
    if profile:
        kwargs["profile"] = profile
    try:
        with AuthContext(**kwargs):
            dsc_model = DataScienceModel.from_id(ocid)
            dsc_model.download_artifact(
                target_dir=artifact_directory,
                force_overwrite=force_overwrite,
                overwrite_existing_artifact=True,
                remove_existing_artifact=True,
                region=region,
                timeout=timeout,
                bucket_uri=bucket_uri,
            )
    except Exception as e:
        print(type(e))
        shutil.rmtree(artifact_directory, ignore_errors=True)
        raise e
