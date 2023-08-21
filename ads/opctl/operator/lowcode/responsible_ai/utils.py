import pandas as pd
import logging
import os


def to_dataframe(scores: dict):
    """convert the score into pandas Dataframe format."""
    scores_dict = {}
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
                                score_for_labels.append(item.get('score'))
                                labels.append("_".join([metric, item.get('label')]) )
                            df_list.append(pd.DataFrame([score_for_labels], columns=labels))
                        df = pd.concat(df_list).reset_index(drop=True)
                    else:
                        logging.debug(score[0][0])
                        raise NotImplemented(f'{score[0][0]} is not a dictionary. Not supported yet.')
                else:
                    df = pd.DataFrame(score, columns=[metric])
            elif isinstance(score, dict):
                df = pd.DataFrame.from_dict(score, orient="index", columns=[metric])
            else:
                df = pd.DataFrame([score], columns=[metric])
        else:
            df = score
        scores_dict[metric] = df
    return scores_dict
