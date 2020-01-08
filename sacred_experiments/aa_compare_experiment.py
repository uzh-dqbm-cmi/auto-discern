from autodiscern import DataManager, model
from autodiscern.experiments.TwoLevelSentenceExperiment import SentenceLevelModelRun
from autodiscern.experiments.Sent2Doc_2ClassProb import Sent2Doc_2ClassProb_Experiment
import pandas as pd
from pathlib import Path
from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sacred.observers import MongoObserver

# run with python sentence_experiment.py

ex = Experiment()
ex.observers.append(MongoObserver.create(
    url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authMechanism=SCRAM-SHA-1',
    db_name='sacred'))


@ex.capture
def get_exp_id(_run):
    return _run._id


@ex.config
def my_config():

    test_mode = False
    if test_mode:
        run_hyperparam_search = False
        num_partitions_to_run = 1
        important_qs = [4]
    else:
        important_qs = [4, 5, 9, 10, 11]
        run_hyperparam_search = True
        num_partitions_to_run = None


    discern_path = "~/switchdrive/Institution/discern"
    cache_file = '2019-08-23_15-10-10_6b9bc88_2019-08-22_15-52-08_6b9bc88_sentence_no_ammend_no_len_1'

    n_estimators_default = 500
    min_samples_leaf_default = 1
    max_depth_default = 100
    max_features_default = 'sqrt'
    class_weight_default = 'balanced_subsample'
    model_class = RandomForestClassifier
    n_estimators = [500]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [100]
    # Minimum number of samples required to split a node
    min_samples_split = [50]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1]
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

    model_run_class = SentenceLevelModelRun
    n_partitions = 5
    stratify_by = 'label'


@ex.automain
def my_main(discern_path, cache_file, important_qs, model_class, hyperparams, model_run_class, n_partitions,
            stratify_by, run_hyperparam_search, num_partitions_to_run):

    # convert hyperparams from a sacred.config.ReadOnlyDict to a regular Dictionary,
    #   otherwise pickling of the experiment fails
    hyperparams_dict = {k: hyperparams[k] for k in hyperparams}

    dm = DataManager(discern_path)
    data_processor = dm.load_cached_data_processor(cache_file)

    # initialize an modeling experiment for each DISCERN question
    model_obj = model_class()
    question_models = {}
    for q_no in important_qs:
        exp_name = "q{}".format(q_no)
        question_models[exp_name] = Sent2Doc_2ClassProb_Experiment(exp_name, data_processor.data, label_key=q_no,
                                                                   model_run_class=model_run_class,
                                                                   model=model_obj,
                                                                   preprocessing_func=data_processor.func,
                                                                   hyperparams=hyperparams_dict,
                                                                   n_partitions=n_partitions,
                                                                   stratify_by=stratify_by, verbose=True)

    # run the experiments
    for q in question_models:
        question_models[q].run(num_partitions_to_run=num_partitions_to_run, run_hyperparam_search=run_hyperparam_search)

    # save F1 and accuracy scores as artifacts
    for metric in ['f1', 'accuracy']:
        all_q_metric = pd.concat([question_models[q].show_evaluation(metric=metric) for q in question_models], axis=0)
        print(all_q_metric)
        all_f1_path = Path(dm.data_path, 'results/{}.txt'.format(metric))
        with open(all_f1_path, 'w') as f:
            f.write(all_q_metric.to_string())
        ex.add_artifact(all_f1_path)

        # save partition accuracies as metric entries?
        for q in all_q_metric.index:
            for col_name in all_q_metric.columns:
                if col_name not in ['mean', 'median', 'stddev']:
                    value = round(all_q_metric.loc[q, col_name], 3)
                    # if the experiment ran with stratified partitions, include the partition id
                    # otherwise, just let whatever partitions be reported in increasing order
                    #   (step is automatically assigned)
                    if 'partition ' in col_name.lower():
                        partiton_number = int(col_name.split(' ')[1])
                        ex.log_scalar("{}_{}".format(q, metric), value, partiton_number)
                    else:
                        ex.log_scalar("{}_{}".format(q, metric), value)

    # save feature importances, hyperparams, and models as artifacts
    for q in question_models:
        print("{}: {}".format(q, model.questions[int(q.split('q')[1])]))
        print(question_models[q].show_sent_feature_importances().head(10))
        sent_feature_path = Path(dm.data_path, 'results/feature_importances_sent_{}.txt'.format(q))
        with open(sent_feature_path, 'w') as f:
            f.write(question_models[q].show_sent_feature_importances().head(20).to_string())
        ex.add_artifact(sent_feature_path)

        print("{}: {}".format(q, model.questions[int(q.split('q')[1])]))
        print(question_models[q].show_doc_feature_importances().head(10))
        doc_feature_path = Path(dm.data_path, 'results/feature_importances_doc_{}.txt'.format(q))
        with open(doc_feature_path, 'w') as f:
            f.write(question_models[q].show_doc_feature_importances().head(20).to_string())
        ex.add_artifact(doc_feature_path)

        sent_hyperparam_path = Path(dm.data_path, 'results/hyperparams_sent_{}.txt'.format(q))
        with open(sent_hyperparam_path, 'w') as f:
            f.write(question_models[q].get_selected_hyperparams(model_level='sent').to_string())
        ex.add_artifact(sent_hyperparam_path)

    # save the models themselves for future inspection
    file_name = '{}.dill'.format(get_exp_id())
    print(type(question_models))
    save_path = dm.save_experiment(question_models, file_name=file_name)
    ex.add_artifact(save_path)
