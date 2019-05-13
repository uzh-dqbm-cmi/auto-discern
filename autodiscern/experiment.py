import random
from sklearn.model_selection import RandomizedSearchCV

from typing import Callable, Dict, List, Tuple


class ExperimentManager:

    def __init__(self, data_dict: Dict, model, hyperparams: Dict, feature_func, n_partitions: int = 5):

        if n_partitions < 2:
            raise ValueError("n_partitions must be greater than 1. Got {}.".format(n_partitions))

        self.data_dict = data_dict
        self.model = model
        self.hyperparams = hyperparams
        self.feature_func = feature_func
        self.n_partitions = n_partitions
        document_ids = list(set([self.data_dict[t]['entity_id'] for t in self.data_dict]))
        self.partitions_by_ids = self.partition_document_ids(document_ids, self.n_partitions)
        self.model_runs = []
        self.all_run_results = []
        self.experiment_results = []

    def run(self):
        self.all_run_results = []
        for i, p in enumerate(self.partitions_by_ids):
            print("Running partition {}...".format(i))
            run_results = self.run_experiment_on_one_partition(data_dict=self.data_dict, partition_ids=p,
                                                               feature_func=self.feature_func,
                                                               model=self.model,
                                                               hyperparams=self.hyperparams)
            self.all_run_results.append(run_results)

        print("Compiling results")
        self.experiment_results = self.summarize_runs(self.all_run_results)
        return self.experiment_results

    def run_experiment_on_one_partition(self, data_dict: Dict, partition_ids: List[int], feature_func: Callable, model,
                                        hyperparams: Dict):
        train_set, test_set = self.materialize_partition(partition_ids, data_dict)
        self.model_runs.append(ModelRun(train_set=train_set, test_set=test_set, model=model, feature_func=feature_func,
                                        hyperparams=hyperparams))
        run_results = self.model_runs[-1].run()
        return run_results

    @classmethod
    def partition_document_ids(cls, doc_list: List[int], n: int) -> List[List[int]]:
        """Randomly shuffle and split the doc_list into n roughly equal lists."""
        random.shuffle(doc_list)
        return [doc_list[i::n] for i in range(n)]

    @classmethod
    def materialize_partition(cls, partition_ids: List[int], data_dict: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Create trainng and testing dataset based on the partition, which indicated the ids for the test set."""

        train_set = [data_dict[d] for d in data_dict if data_dict[d]['entity_id'] not in partition_ids]
        test_set = [data_dict[d] for d in data_dict if data_dict[d]['entity_id'] in partition_ids]

        return train_set, test_set

    @classmethod
    def summarize_runs(cls, run_results):
        return run_results


class ModelRun:

    def __init__(self, train_set: List[Dict], test_set: List[Dict], feature_func: Callable, model, hyperparams: Dict):
        self.train_set = train_set
        self.test_set = test_set
        self.feature_func = feature_func
        self.model = model
        self.hyperparams = hyperparams

        self.feature_cols = []
        self.encoders = {}

    def run(self):
        self.x_train, self.x_test, self.y_train, self.y_test, self.feature_cols, self.encoders = self.feature_func(
            self.train_set, self.test_set)
        # self.model = self.search_hyperparameters(self.model, self.hyperparams, x_train, y_train)
        self.model.fit(self.x_train, self.y_train)
        self.evaluation = self.evaluate_model(self.model, self.x_test, self.y_test)
        return self.evaluation

    @classmethod
    def search_hyperparameters(cls, model, hyperparams, x_train, y_train):
        model = RandomizedSearchCV(estimator=model, param_distributions=hyperparams, n_iter=5, cv=1, verbose=2,
                                   random_state=42, n_jobs=-1)
        # Fit the random search model
        model.fit(x_train, y_train)
        return model

    @classmethod
    def evaluate_model(cls, model, x_test, y_test):
        score = model.score(x_test, y_test)
        return score
