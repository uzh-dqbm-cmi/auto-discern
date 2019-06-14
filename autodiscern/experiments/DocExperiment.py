import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
import autodiscern.experiment as ade


class DocExperiment(ade.PartitionedExperiment):

    @classmethod
    def run_experiment_on_one_partition(cls, data_dict: Dict, label_key: str, partition_ids: List[int], model,
                                        hyperparams: Dict, encoders: Dict, skip_hyperparam_search: bool):

        train_set, test_set = cls.materialize_partition(partition_ids, data_dict)

        mr = DocLevelModelRun(train_set=train_set, test_set=test_set, label_key=label_key, model=model,
                              hyperparams=hyperparams, encoders=encoders)
        mr.run(skip_hyperparam_search=skip_hyperparam_search)
        return mr


class DocLevelModelRun(ade.ModelRun):

    @classmethod
    def train_encoders(cls, train_set: List[Dict]):
        corpus_train = [entity_dict['content'] for entity_dict in train_set]
        vectorizer = TfidfVectorizer(max_df=0.9999, min_df=0.0001, max_features=200, stop_words='english')
        vectorizer.fit(corpus_train)
        encoders = {
            'vectorizer': vectorizer,
        }
        return encoders

    @classmethod
    def build_x_features(cls, data_set: List[Dict], encoders):
        corpus = [entity_dict['content'] for entity_dict in data_set]
        x_tfidf = encoders['vectorizer'].transform(corpus)
        feature_vec = pd.concat([entity_dict['feature_vec'] for entity_dict in data_set], axis=0)
        x_all = hstack([x_tfidf, coo_matrix(feature_vec)])

        feature_cols = encoders['vectorizer'].get_feature_names()
        feature_cols.extend(feature_vec.columns)

        return x_all, feature_cols

    # @classmethod
    # def build_y_vector(cls, data_set, label_key):
    #     return [entity_dict[label_key] for entity_dict in data_set]
    #
    # @classmethod
    # def build_data(cls, train_set: List[Dict], test_set: List[Dict], label_key: str, encoders: Dict) -> \
    #         Tuple[coo_matrix, coo_matrix, List, List, List, Dict]:
    #
    #     encoders = cls.train_encoders(train_set)
    #
    #     x_train, feature_cols = cls.build_x_features(train_set, encoders)
    #     x_test, feature_cols = cls.build_x_features(test_set, encoders)
    #
    #     y_train = cls.build_y_vector(train_set, label_key)
    #     y_test = cls.build_y_vector(test_set, label_key)
    #
    #     return x_train, x_test, y_train, y_test, feature_cols, encoders
