import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple

import autodiscern.experiment as ade


class SentenceLevelModelRun(ade.ModelRun):
    data_dict = {}
    doc_level_info = {}
    train_set_upd = None
    test_set_upd = None

    def __init__(self, train_set: List[Dict], test_set: List[Dict], model, hyperparams: Dict, 
                 data_dict: Dict, doc_level_info: Dict):
        super().__init__(train_set, test_set, model, hyperparams)
        SentenceLevelModelRun._assign_vars(data_dict, doc_level_info)

    @classmethod
    def _assign_vars(cls, data_dict, doc_level_info):
        cls.data_dict = data_dict
        cls.doc_level_info = doc_level_info
        print('assigning cls vars in SentenceLevelModelRun')
        print('len(data_dict): ', len(cls.data_dict))
        print('len(doc_level_info): ', len(cls.doc_level_info))

    @classmethod
    def _extract_dset(cls, doc_ids, doc_level_info, data_dict):
        dset = []
        for doc_id in doc_ids:
            num_sents = doc_level_info[doc_id]['num_sent']
            for i in range(0, num_sents):
                dset.append(data_dict['{}-{}'.format(doc_id, i)])
        return dset

    @classmethod
    def _inflate_tfidf_doc_features(cls, x_tfidf, doc_ids, doc_level_info):
        arr_reps = x_tfidf.toarray()
        # print('arr_reps.shape: ', arr_reps.shape)
        repeat_info = [doc_level_info[doc_id]['num_sent'] for doc_id in doc_ids]
        # print('repeat_info: ', repeat_info)
        inflated_reps = np.repeat(arr_reps, repeat_info, axis=0)
        x_upd_tfidf = coo_matrix(inflated_reps)
        return x_upd_tfidf

    @classmethod
    def build_features(cls, train_set: List[Dict], test_set: List[Dict]) -> Tuple[coo_matrix, coo_matrix, List, List,
                                                                                  List, Dict]:
                                                           
        doc_level_info = SentenceLevelModelRun.doc_level_info
        document_ids = list(doc_level_info.keys())
        document_ids.sort()

        data_dict = SentenceLevelModelRun.data_dict
        train_ids = list(set([int(entity_dict['entity_id']) for entity_dict in train_set]))
        test_ids = list(set([int(entity_dict['entity_id']) for entity_dict in test_set]))
        
        train_ids.sort()
        test_ids.sort()
        
        print('build_features method..')
        print(train_ids)
        print(test_ids)
        print('len(doc_level_info): ', len(doc_level_info))

        corpus_train = [doc_level_info[doc_id]['content'] for doc_id in train_ids]
        corpus_test = [doc_level_info[doc_id]['content'] for doc_id in test_ids]
        
        # update train set and test set lists to reflect the above order
        train_set = cls._extract_dset(train_ids, doc_level_info, data_dict)
        test_set = cls._extract_dset(test_ids, doc_level_info, data_dict)
        
        # set the updated train/test sets
        cls.train_set_upd = train_set
        cls.test_set_upd = test_set

        feature_vec_train = pd.concat([entity_dict['feature_vec'] for entity_dict in train_set], axis=0)
        feature_vec_test = pd.concat([entity_dict['feature_vec'] for entity_dict in test_set], axis=0)

        y_train = [entity_dict['label'] for entity_dict in train_set]
        y_test = [entity_dict['label'] for entity_dict in test_set]

        # tfidf using doc level info
        vectorizer = TfidfVectorizer(max_df=0.98, min_df=0.01, stop_words='english')
        x_train_tfidf = vectorizer.fit_transform(corpus_train)
        x_test_tfidf = vectorizer.transform(corpus_test)

        x_train_tfidf = cls._inflate_tfidf_doc_features(x_train_tfidf, train_ids, doc_level_info)
        x_test_tfidf = cls._inflate_tfidf_doc_features(x_test_tfidf, test_ids, doc_level_info)

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
        # turn string labels into numbers
        train_set['pred_num'] = np.where(train_set['sub_prediction'] == 'positive', 2,
                                         np.where(train_set['sub_prediction'] == 'neutral', 1, 0))
        train_set['label_num'] = np.where(train_set['label'] == 'positive', 2,
                                          np.where(train_set['label'] == 'neutral', 1, 0))
        test_set['pred_num'] = np.where(test_set['sub_prediction'] == 'positive', 2,
                                        np.where(test_set['sub_prediction'] == 'neutral', 1, 0))
        test_set['label_num'] = np.where(test_set['label'] == 'positive', 2,
                                         np.where(test_set['label'] == 'neutral', 1, 0))

        # generate a df where each row represents a document (the document is partitioned into 10 equal parts)
        # such that we compute average prediction on each part to form a feature (i.e. a column) in df
        # hence for each row, we have 10 features
        x_train = sents_to_doc_buckets_mean(train_set)
        x_test = sents_to_doc_buckets_mean(test_set)

        y_train = train_set.groupby('doc_id')['label_num'].median()
        y_test = test_set.groupby('doc_id')['label_num'].median()

        feature_cols = x_train.columns
        encoders = {}

        return x_train, x_test, y_train, y_test, feature_cols, encoders


def sents_to_doc_buckets_mean(df):
    series_of_list_of_bucket_avgs = df.groupby('doc_id')['pred_num'].apply(calc_avg_over_ten_parts)
    df = series_of_list_to_df_columns(series_of_list_of_bucket_avgs)
    return df


def mean_0_when_empty(p):
    if len(p) == 0:
        return 0
    else:
        return sum(p)/len(p)


def calc_avg_over_equal_parts(the_list, n_partitions):
    n = int(np.ceil(len(the_list)/n_partitions))
    partitions = [the_list[i*n:i*n+n] for i in range(n_partitions)]
    return [mean_0_when_empty(p) for p in partitions]


def calc_avg_over_ten_parts(the_list):
    return calc_avg_over_equal_parts(the_list, 10)


def series_of_list_to_df_columns(a_series_of_lists):
    df = pd.DataFrame()
    for ix in a_series_of_lists.index:
        d = {}
        #     d['doc_id'] = ix
        for col_i, value in enumerate(a_series_of_lists[ix]):
            d[col_i] = value
        df = pd.concat([df, pd.DataFrame(d, index=[ix])])
    return df
