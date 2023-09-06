import pandas as pd
import numpy as np
import logging
import subprocess
import oci
from .guardrail.llm import LLMEndpoint, MDEndpoint, OCIEndpoint
from typing import Union


def to_dataframe(scores: Union[dict, pd.DataFrame]):
    """convert the score into pandas Dataframe format."""
    # scores_dict = {}
    if isinstance(scores, pd.DataFrame):
        return scores
    for metric, score in scores.items():
        if not isinstance(score, pd.DataFrame):
            if isinstance(score, list):
                if isinstance(score[0], list):
                    if isinstance(score[0][0], dict):
                        df_list = []
                        for each_score in score:
                            labels = []
                            score_for_labels = []
                            for item in each_score:
                                score_for_labels.append(item.get("score"))
                                labels.append("_".join([metric, item.get("label")]))
                            df_list.append(
                                pd.DataFrame([score_for_labels], columns=labels)
                            )
                        df = pd.concat(df_list).reset_index(drop=True)
                    else:
                        logging.debug(score[0][0])
                        raise NotImplemented(
                            f"{score[0][0]} is not a dictionary. Not supported yet."
                        )
                else:
                    df = pd.DataFrame(score, columns=[metric])
            elif isinstance(score, dict):
                df = pd.DataFrame.from_dict(score, orient="index", columns=[metric])
            else:
                df = pd.DataFrame([score], columns=[metric])
        else:
            df = score
        # scores_dict[metric] = df
    return df


def postprocess_sentence_level_dataframe(df):
    columns = [
        col for col in df.columns if col not in ["index", "text", "references"]
    ]
    scores = []
    for col in columns:
        labels = []
        starting_pos = 0
        sents = []
        label = []
        prev_idx = 0
        pred_level_sents = []
        for i, row in df.iterrows():
            if row["index"] != prev_idx:
                pred_level_sents.append(" ".join(sents))
                labels.append(label)
                prev_idx = row["index"]
                sents = []
                label = []
                starting_pos = 0

            l = len(row["text"])
            label.append((starting_pos, l, row[col]))
            starting_pos += l + 1
            sents.append(row["text"])

        if row["index"] == prev_idx:
            pred_level_sents.append(" ".join(sents))
            labels.append(label)
        scores.append(labels)
    df_final = pd.DataFrame(scores + [pred_level_sents]).T
    df_final.columns = columns + ["text"]
    return df_final


def authenticate_with_security_token(profile):
    return subprocess.check_output(
        f"oci session authenticate --profile-name {profile} --region us-ashburn-1 --tenancy-name bmc_operator_access",
        shell=True,
    )


def get_oci_auth(profile):
    oci_config = oci.config.from_file(profile_name=profile)
    if "security_token_file" in oci_config and "key_file" in oci_config:
        token_file = oci_config["security_token_file"]
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read()
        private_key = oci.signer.load_private_key_from_file(oci_config["key_file"])
        signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
        oci_auth = {"config": oci_config, "signer": signer}
        return oci_auth
    else:
        oci_auth = {"config": oci_config}
        return oci_auth


model_endpoint_mapping = {
    "cohere": {
        "profile": "DEFAULT",
        "endpoint": "https://generativeai-dev.aiservice.us-chicago-1.oci.oraclecloud.com",
    },
    "llama7b": {
        "profile": "custboat",
        "endpoint": "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.iad.amaaaaaay75uckqay7so6w2bpwreqxisognml72kdqi4qcjdtnpfykh4xtsq/predict",
    },
    "llama13b": {
        "profile": "custboat",
        "endpoint": "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.iad.amaaaaaay75uckqaj5a53ebpi2zutlf733n22us2lgycd4xesvsn6pzecisa/predict",
    },
}

import subprocess


def authenticate_with_security_token(profile):
    return subprocess.check_output(
        f"oci session authenticate --profile-name {profile} --region us-ashburn-1 --tenancy-name bmc_operator_access",
        shell=True,
    )


def init_endpoint(name: str):
    profile = model_endpoint_mapping[name]["profile"]
    endpoint = model_endpoint_mapping[name]["endpoint"]
    if name == "cohere":
        oci_config = oci.config.from_file(profile_name=profile)
        return OCIEndpoint(
            compartment_id=oci_config["compartment_id"],
            service_endpoint=endpoint,
            config=oci_config,
        )
    elif name in ["llama7b", "llama13b"]:
        authenticate_with_security_token(profile)
        oci_auth = get_oci_auth(profile)
        return MDEndpoint(endpoint=endpoint, **oci_auth)
    else:
        raise NotImplementedError(f"Model {name} not implemented.")


def apply_filter(score: pd.DataFrame, threshold: float, direction: str = "<="):
    filters = []
    for col in score.columns:
        if col != "text":
            if direction == "<=":
                filters.append((score[col] <= threshold).values)
            else:
                filters.append((score[col] >= threshold).values)
    return np.logical_and(*filters) if len(filters) > 1 else filters[0]
