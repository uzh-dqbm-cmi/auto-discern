import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from autodiscern import experiment, model, lemmatization


# class DocExperiment(experiment.PartitionedExperiment):
#
#     @classmethod
#     def run_experiment_on_one_partition(cls, data_dict: Dict, label_key: str, partition_ids: List[int],
#                                         preprocessing_func: Callable, model, hyperparams: Dict,
#                                         skip_hyperparam_search: bool):
#
#         train_set, test_set = cls.materialize_partition(partition_ids, data_dict)
#
#         mr = DocLevelModelRun(train_set=train_set, test_set=test_set, label_key=label_key,
#                               preprocessing_func=preprocessing_func, model=model, hyperparams=hyperparams)
#         mr.run(skip_hyperparam_search=skip_hyperparam_search)
#         return mr


class DocLevelModelRun(experiment.ModelRun):

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

        lemmatizer = lemmatization.Lemmatizer()
        corpus_lemmas = lemmatizer.lemmatize_list_of_texts(corpus)

        x_tfidf = encoders['vectorizer'].transform(corpus_lemmas)
        feature_vec = pd.concat([entity_dict['feature_vec'] for entity_dict in data_set], axis=0)
        x_all = hstack([x_tfidf, coo_matrix(feature_vec)])

        feature_cols = encoders['vectorizer'].get_feature_names()
        feature_cols.extend(feature_vec.columns)

        return x_all, feature_cols

    @classmethod
    def build_y_vector(cls, data_set: List[Dict], label_key: str) -> List:
        """
        Extract the labels from each data dict and compile into one y vector.
        Args:
            data_set: List of data dicts.
            label_key: The key int he data dicts under which the label is stored.

        Returns:
            Array-type
        """
        return [model.zero_one_category(model.get_score_for_question(entity_dict, label_key)) for entity_dict in
                data_set]
