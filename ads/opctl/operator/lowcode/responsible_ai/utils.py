import pandas as pd
import logging
import subprocess
import oci


def to_dataframe(scores: dict):
    """convert the score into pandas Dataframe format."""
    # scores_dict = {}
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
        col for col in df.columns if col not in ["index", "predictions", "references"]
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

            l = len(row["predictions"])
            label.append((starting_pos, l, row[col]))
            starting_pos += l + 1
            sents.append(row["predictions"])

        if row["index"] == prev_idx:
            pred_level_sents.append(" ".join(sents))
            labels.append(label)
        scores.append(labels)
    df_final = pd.DataFrame(scores + [pred_level_sents]).T
    df_final.columns = columns + ["predictions"]
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