import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple
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
    def build_features(cls, train_set: List[Dict], test_set: List[Dict], label_key: str, encoders: Dict) -> \
            Tuple[coo_matrix, coo_matrix, List, List, List, Dict]:
        corpus_train = [entity_dict['content'] for entity_dict in train_set]
        corpus_test = [entity_dict['content'] for entity_dict in test_set]

        feature_vec_train = pd.concat([entity_dict['feature_vec'] for entity_dict in train_set], axis=0)
        feature_vec_test = pd.concat([entity_dict['feature_vec'] for entity_dict in test_set], axis=0)

        vectorizer = TfidfVectorizer(max_df=0.9999, min_df=0.0001, stop_words='english')
        x_train_tfidf = vectorizer.fit_transform(corpus_train)
        x_test_tfidf = vectorizer.transform(corpus_test)

        x_train = hstack([x_train_tfidf, coo_matrix(feature_vec_train)])
        x_test = hstack([x_test_tfidf, coo_matrix(feature_vec_test)])

        y_train = [entity_dict[label_key] for entity_dict in train_set]
        y_test = [entity_dict[label_key] for entity_dict in test_set]

        feature_cols = vectorizer.get_feature_names()
        feature_cols.extend(feature_vec_train.columns)
        encoders = {'vectorizer': vectorizer}

        return x_train, x_test, y_train, y_test, feature_cols, encoders
