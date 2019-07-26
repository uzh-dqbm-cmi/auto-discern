from autodiscern import DataManager, model
from autodiscern.experiment import PartitionedExperiment
from autodiscern.experiments.DocExperiment import DocLevelModelRun
import pandas as pd
from pathlib import Path
from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sacred.observers import MongoObserver

# run with python first_experiment.py

ex = Experiment()
ex.observers.append(MongoObserver.create(
    url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authMechanism=SCRAM-SHA-1',
    db_name='sacred'))


@ex.config
def my_config():
    discern_path = "~/switchdrive/Institution/discern"
    cache_file = '2019-07-24_15-43-08_bd60a7f_with_mm_and_ner_ammendments.pkl'
    # cache_file = '2019-07-24_13-40-50_166c23e_test_mm_and_ner_ammendments_on_subset_30'
    important_qs = [4, 5, 9, 10, 11]
    
    n_estimators_default = 500
    min_samples_leaf_default = 5
    max_depth_default = 50
    max_features_default = 'sqrt'
    class_weight_default = 'balanced_subsample'
    model_class = RandomForestClassifier
    n_estimators = [50, 100, 500]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [10, 50, 100]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 50, 500]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [3, 5, 10, 100]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # Create the random grid
    hyperparams = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   # 'bootstrap': bootstrap,
                   'class_weight': ['balanced_subsample'],
                   }
    
    model_run_class = DocLevelModelRun
    n_partitions = 5
    stratified = True


@ex.automain
def my_main(discern_path, cache_file, important_qs, model_class, hyperparams, model_run_class, n_partitions, stratified):

    dm = DataManager(discern_path)
    data_dict, processing_func = dm.load_cached_data_processor(cache_file)

    # initialize an modeling experiment for each DISCERN question
    model_obj = model_class()
    question_models = {}
    for q_no in important_qs:
        exp_name = "q{}".format(q_no)
        question_models[exp_name] = PartitionedExperiment(exp_name, data_dict, label_key=q_no,
                                                          model_run_class=model_run_class,
                                                          model=model_obj, preprocessing_func=processing_func,
                                                          hyperparams=hyperparams, n_partitions=n_partitions,
                                                          stratified=stratified, verbose=True)

    # run the experiments
    for q in question_models:
        question_models[q].run()

    # view F1 scores
    all_q_f1 = pd.concat([question_models[q].show_evaluation(metric='f1') for q in question_models], axis=0)
    print(all_q_f1)
    all_f1_path = Path(dm.data_path, 'results/f1.txt')
    with open(all_f1_path, 'w') as f:
        f.write(all_q_f1.to_string())
    ex.add_artifact(all_f1_path)

    # try saving partition accuracies as metric entries?

    # try saving individual f1 scores as metrics
    for q in all_q_f1.index:
        for col_name in all_q_f1.columns:
            if 'partition' in col_name:
                partiton_number = int(col_name.split(' ')[1])
                value = round(all_q_f1.loc[q, col_name], 3)
                ex.log_scalar("{}_f1".format(q), value, partiton_number)

    # save the result
    ex.result = all_q_f1['median'].to_string()

    # view feature importances
    for q in question_models:
        print("{}: {}".format(q, model.questions[int(q.split('q')[1])]))
        print(question_models[q].show_feature_importances().head(10))
        feature_path = Path(dm.data_path, 'results/feature_importances_{}.txt'.format(q))
        with open(feature_path, 'w') as f:
            f.write(question_models[q].show_feature_importances().head(20).to_string())
        ex.add_artifact(feature_path)
