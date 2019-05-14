import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple

import autodiscern.experiment as ade


class TwoLevelSentenceExperiment(ade.ExperimentManager):

    @classmethod
    def run_experiment_on_one_partition(cls, data_dict: Dict, partition_ids: List[int], model, hyperparams: Dict):

        train_set, test_set = cls.materialize_partition(partition_ids, data_dict)

        # run SentenceLevelModel
        sl_mr = SentenceLevelModelRun(train_set=train_set, test_set=test_set, model=model, hyperparams=hyperparams)
        sl_mr.run()

        # use predictions from SentenceLevelModel to create training set for SentenceToDocModel
        data_set_train = cls.create_sent_to_doc_data_set(sl_mr.model, sl_mr.x_train, sl_mr.train_set)
        data_set_test = cls.create_sent_to_doc_data_set(sl_mr.model, sl_mr.x_test, sl_mr.test_set)

        dl_mr = SentenceToDocModelRun(train_set=data_set_train, test_set=data_set_test, model=model,
                                      hyperparams=hyperparams)
        dl_mr.run()

        return {'sentence_level': sl_mr,
                'doc_level': dl_mr}

    @classmethod
    def create_sent_to_doc_data_set(cls, model, x_feature_set, data_set: List):
        cat_prediction = model.predict(x_feature_set)
        proba_prediction = model.predict_proba(x_feature_set)
        new_data_set = pd.DataFrame({
            'doc_id': [d['entity_id'] for d in data_set],
            'sub_id': [d['sub_id'] for d in data_set],
            'sub_prediction': cat_prediction,
            'proba_0': [i[0] for i in proba_prediction],
            'proba_1': [i[1] for i in proba_prediction],
            'proba_2': [i[2] for i in proba_prediction],
            'label': [d['label'] for d in data_set],
        })
        return new_data_set


class SentenceLevelModelRun(ade.ModelRun):

    @classmethod
    def build_features(cls, train_set: List[Dict], test_set: List[Dict]) -> Tuple[coo_matrix, coo_matrix, List, List,
                                                                                  List, Dict]:
        corpus_train = [entity_dict['content'] for entity_dict in train_set]
        corpus_test = [entity_dict['content'] for entity_dict in test_set]

        feature_vec_train = pd.concat([entity_dict['feature_vec'] for entity_dict in train_set], axis=0)
        feature_vec_test = pd.concat([entity_dict['feature_vec'] for entity_dict in test_set], axis=0)

        y_train = [entity_dict['label'] for entity_dict in train_set]
        y_test = [entity_dict['label'] for entity_dict in test_set]

        vectorizer = TfidfVectorizer(max_df=0.9999, min_df=0.0001)
        x_train_tfidf = vectorizer.fit_transform(corpus_train)
        x_test_tfidf = vectorizer.transform(corpus_test)

        x_train = hstack([x_train_tfidf, coo_matrix(feature_vec_train)])
        x_test = hstack([x_test_tfidf, coo_matrix(feature_vec_test)])
        feature_cols = vectorizer.get_feature_names()
        feature_cols.extend(feature_vec_train.columns)
        encoders = {'vectorizer': vectorizer}

        return x_train, x_test, y_train, y_test, feature_cols, encoders


class SentenceToDocModelRun(ade.ModelRun):

    @classmethod
    def build_features(cls, train_set: pd.DataFrame, test_set: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List,
                                                                                      List, List, Dict]:
        na_number = -1

        # turn string labels into numbers
        train_set['pred_num'] = np.where(train_set['sub_prediction'] == 'positive', 2,
                                         np.where(train_set['sub_prediction'] == 'neutral', 1, 0))
        train_set['label_num'] = np.where(train_set['label'] == 'positive', 2,
                                          np.where(train_set['label'] == 'neutral', 1, 0))
        test_set['pred_num'] = np.where(test_set['sub_prediction'] == 'positive', 2,
                                        np.where(test_set['sub_prediction'] == 'neutral', 1, 0))
        test_set['label_num'] = np.where(test_set['label'] == 'positive', 2,
                                         np.where(test_set['label'] == 'neutral', 1, 0))

        # compile results by document
        x_train = pd.pivot_table(train_set, index='doc_id', columns='sub_id',  values='pred_num', aggfunc='median'
                                 ).fillna(na_number)
        x_test = pd.pivot_table(test_set, index='doc_id', columns='sub_id', values='pred_num', aggfunc='median'
                                ).fillna(na_number)
        # make sure x_test has the same columns as x_train
        cols_to_add_to_x_test = [col for col in x_train.columns if col not in x_test.columns]
        for col in cols_to_add_to_x_test:
            x_test[col] = na_number
        x_test = x_test[x_train.columns]

        y_train = train_set.groupby('doc_id')['label_num'].median()
        y_test = test_set.groupby('doc_id')['label_num'].median()

        feature_cols = x_train.columns
        encoders = {}

        return x_train, x_test, y_train, y_test, feature_cols, encoders


def calc_avg_over_equal_parts(the_list, n_partitions):
    n = int(np.ceil(len(the_list)/n_partitions))
    partitions = [the_list[i*n:i*n+n] for i in range(n_partitions)]
    return [sum(p)/len(p) for p in partitions]
