from abc import ABC, abstractmethod


class OnlineFeatureStoreStrategy(ABC):
    """
    Abstract base class for defining online feature store strategies.
    Online strategies facilitate read and write operations for feature groups or datasets in an online serving environment.
    """

    @abstractmethod
    def write(self, feature_group_or_dataset, feature_group_or_dataset_job, dataframe):
        """
        Abstract method for writing data to the online feature store.

        Parameters:
            - feature_group_or_dataset (FeatureGroup or Dataset): The feature group or dataset to write to.
            - feature_group_or_dataset_job (FeatureGroupJob or DatasetJob): The job associated with the feature group or dataset.
            - dataframe (DataFrame): The data to be written to the online feature store.

        Returns:
            None
        """
        pass

    @abstractmethod
    def read(self, feature_group_or_dataset, primary_key_vector):
        """
        Abstract method for reading data from the online feature store.

        Parameters:
            - feature_group_or_dataset (FeatureGroup or Dataset): The feature group or dataset to read from.
            - primary_key_vector (list): The primary key vector used to retrieve specific records.

        """
        pass

    def get_embedding_vector(
        self,
        feature_group_or_dataset,
        embedding_field,
        k_neighbors,
        query_embedding_vector,
        max_candidate_pool,
    ):
        """
        Placeholder method for obtaining embedding vectors in an online serving environment.

        Parameters:
            - feature_group_or_dataset (FeatureGroup or Dataset): The feature group or dataset to retrieve embeddings from.
            - embedding_field (str): The field containing embedding vectors.
            - k_neighbors (int): The number of neighbors to consider.
            - query_embedding_vector (list): The embedding vector for which to find neighbors.
            - max_candidate_pool (int): The maximum number of candidates to consider.
        """
        pass
