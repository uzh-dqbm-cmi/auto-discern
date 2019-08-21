import numpy as np
import pandas as pd
from typing import Dict, List, Callable

import autodiscern.experiment as ade
from autodiscern import model as admodel
from autodiscern.experiments.TwoLevelSentenceExperiment import SentenceLevelModelRun


class Sent2Doc_2ClassProb_Experiment(ade.PartitionedExperiment):

    @classmethod
    def run_experiment_on_one_partition(cls, data_dict: Dict, label_key: str, partition_ids: List[int],
                                        preprocessing_func: Callable, model_run_class: ade.ModelRun, model,
                                        hyperparams: Dict, run_hyperparam_search: bool):

        train_set, test_set = cls.materialize_partition(partition_ids, data_dict)

        # run SentenceLevelModel
        sl_mr = SentenceLevelModelRun(train_set=train_set, test_set=test_set, label_key=label_key, model=model,
                                      preprocessing_func=preprocessing_func, hyperparams=hyperparams)
        sl_mr.run(run_hyperparam_search=run_hyperparam_search)

        # use predictions from SentenceLevelModel to create training set for SentenceToDocModel
        doc_set_train = cls.create_sent_to_doc_data_set(sl_mr.model, sl_mr.x_train, sl_mr.train_set, label_key)
        doc_set_test = cls.create_sent_to_doc_data_set(sl_mr.model, sl_mr.x_test, sl_mr.test_set, label_key)

        dl_mr = SentenceToDocProbaModelRun(train_set=doc_set_train, test_set=doc_set_test, label_key=label_key,
                                           model=model, preprocessing_func=preprocessing_func, hyperparams=hyperparams)
        dl_mr.run(run_hyperparam_search=run_hyperparam_search)

        return {'sentence_level': sl_mr,
                'doc_level': dl_mr}

    @classmethod
    def create_sent_to_doc_data_set(cls, model, x_feature_set, data_set: List, label_key: str):
        cat_prediction = model.predict(x_feature_set)
        proba_prediction = model.predict_proba(x_feature_set)
        new_data_set = pd.DataFrame({
            'doc_id': [d['entity_id'] for d in data_set],
            'sub_id': [d['sub_id'] for d in data_set],
            'sub_prediction': cat_prediction,
            'proba_0': [i[0] for i in proba_prediction],
            'proba_1': [i[1] for i in proba_prediction],
            'label': [admodel.zero_one_category(admodel.get_score_for_question(d, label_key)) for d in data_set],
        })
        return new_data_set

    def show_feature_importances(self):
        print("ERROR: this is not a valid function in `Sent2Doc_2ClassProb_Experiment`."
              "Use `show_sent_feature_importances` or `show_doc_feature_importances`.")

    def show_sent_feature_importances(self):
        all_feature_importances = pd.DataFrame()
        for partition_id in self.model_runs:
            partition_feature_importances = self.model_runs[partition_id]['sentence_level'].get_feature_importances()
            partition_feature_importances.columns = [partition_id]
            all_feature_importances = pd.merge(all_feature_importances, partition_feature_importances, how='outer',
                                               left_index=True, right_index=True)
        all_feature_importances['median'] = all_feature_importances.median(axis=1)
        return all_feature_importances.sort_values('median', ascending=False)

    def show_doc_feature_importances(self):
        all_feature_importances = pd.DataFrame()
        for partition_id in self.model_runs:
            partition_feature_importances = self.model_runs[partition_id]['doc_level'].get_feature_importances()
            partition_feature_importances.columns = [partition_id]
            all_feature_importances = pd.merge(all_feature_importances, partition_feature_importances, how='outer',
                                               left_index=True, right_index=True)
        all_feature_importances['median'] = all_feature_importances.median(axis=1)
        return all_feature_importances.sort_values('median', ascending=False)

    def show_evaluation(self, metric: str = 'accuracy'):
        all_accuracy = {}
        for partition_id in self.model_runs:
            all_accuracy[partition_id] = self.model_runs[partition_id]['doc_level'].evaluation[metric]
        all_accuracy_df = pd.DataFrame(all_accuracy, index=[self.name])
        median = all_accuracy_df.median(axis=1)
        stddev = all_accuracy_df.std(axis=1)
        all_accuracy_df['median'] = median
        all_accuracy_df['stddev'] = stddev
        return all_accuracy_df.sort_values('median', ascending=False)

    def generate_predictor(self, partition=0):
        """
        Return a Predictor from the trained model of a specific partition.

        Args:
            partition: The partition id of the model to return. Defaults to 0.

        Returns: a Predictor.

        """
        # TODO: THIS WILL NOT WORK AS IS!!
        # def make_prediction(input):
        #     sent_predictor = self.model_runs['sentence_level'].generate_predictor()
        #     doc_predictor = self.model_runs['doc_level'].generate_predictor()
        #     return doc_predictor(sent_predictor(input))

        return self.model_runs[partition].generate_predictor()


class SentenceToDocProbaModelRun(ade.ModelRun):

    @classmethod
    def train_encoders(cls, train_set: List[Dict]):
        return None

    @classmethod
    def build_x_features(cls, data_set: pd.DataFrame, encoders: Dict):
        x = cls.sents_to_doc_buckets_mean(data_set)
        feature_cols = x.columns
        return x, feature_cols

    @classmethod
    def build_y_vector(cls, data_set: pd.DataFrame, label_key: str) -> List:
        return data_set.groupby('doc_id')['label'].median()

    @classmethod
    def sents_to_doc_buckets_mean(cls, df: pd.DataFrame):
        series_of_list_of_bucket_avgs = df.groupby('doc_id')['sub_prediction'].apply(calc_avg_over_ten_parts)
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
