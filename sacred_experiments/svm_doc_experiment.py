from autodiscern import DataManager, model
from autodiscern.experiment import PartitionedExperiment
from autodiscern.experiments.DocExperiment import DocLevelModelRun
import pandas as pd
from pathlib import Path
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.svm import SVC

# run with python svm_doc_experiment.py
# or python svm_doc_experiment.py with "test_mode=True"

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
    cache_file = '2019-08-21_18-56-53_fae71bc_doc.pkl'
    # cache_file = '2019-07-24_15-43-08_bd60a7f_with_mm_and_ner_ammendments.pkl'
    # cache_file = '2019-07-24_13-40-50_166c23e_test_mm_and_ner_ammendments_on_subset_30'

    model_class = SVC

    hyperparams = {
                   'C': [0.000001, 0.00001, 0.0001, 0.01, 1],
                   'kernel': ['rbf', 'poly', 'sigmoid'],
                   'gamma': ['scale', 0.1, 0.01, 0.001, 0.0001],
                   'class_weight': ['balanced'],
                   }
    
    model_run_class = DocLevelModelRun
    n_partitions = 5
    stratify_by = 'label'
    feature_subset = None


@ex.automain
def my_main(discern_path, cache_file, important_qs, model_class, hyperparams, model_run_class, n_partitions,
            stratify_by, run_hyperparam_search, num_partitions_to_run, feature_subset, test_mode):

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
        question_models[exp_name] = PartitionedExperiment(exp_name, data_processor.data, label_key=q_no,
                                                          model_run_class=model_run_class,
                                                          model=model_obj, preprocessing_func=data_processor.func,
                                                          hyperparams=hyperparams_dict, n_partitions=n_partitions,
                                                          stratify_by=stratify_by, feature_subset = feature_subset,
                                                          verbose=True)

    # run the experiments
    for q in question_models:
        question_models[q].run(num_partitions_to_run=num_partitions_to_run, run_hyperparam_search=run_hyperparam_search)

        # SVM does not have feature importances!
        # save feature importances as artifacts
        # print("{}: {}".format(q, model.questions[int(q.split('q')[1])]))
        # print(question_models[q].show_feature_importances().head(10))
        # feature_path = Path(dm.data_path, 'results/{}_feature_importances.txt'.format(q))
        # with open(feature_path, 'w') as f:
        #     f.write(question_models[q].show_feature_importances().head(20).to_string())
        # ex.add_artifact(feature_path)

        # save hyperparams
        sent_hyperparam_path = Path(dm.data_path, 'results/{}_hyperparams.txt'.format(q))
        with open(sent_hyperparam_path, 'w') as f:
            f.write(question_models[q].get_selected_hyperparams().to_string())
        ex.add_artifact(sent_hyperparam_path)

        # save confusion matrix, test set
        y_true = []
        y_pred = []
        for p in question_models[q].model_runs:
            y_true.extend(question_models[q].model_runs[p].y_test)
            y_pred.extend(question_models[q].model_runs[p].y_test_predicted)
        classes = question_models[q].model_runs[p].model.classes_

        conf_matrix_path = Path(dm.data_path, 'results/{}_confusion_matrix_test.png'.format(q))
        title = "{} Test Set".format(q)
        plot = model.plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=True, title=title)
        fig = plot.get_figure()
        fig.savefig(conf_matrix_path)
        ex.add_artifact(conf_matrix_path)

        # save confusion matrix, train set
        y_true = []
        y_pred = []
        for p in question_models[q].model_runs:
            y_true.extend(question_models[q].model_runs[p].y_train)
            y_pred.extend(question_models[q].model_runs[p].y_train_predicted)
        classes = question_models[q].model_runs[p].model.classes_

        conf_matrix_path = Path(dm.data_path, 'results/{}_confusion_matrix_train.png'.format(q))
        title = "{} Train Set".format(q)
        plot = model.plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=True, title=title)
        fig = plot.get_figure()
        fig.savefig(conf_matrix_path)
        ex.add_artifact(conf_matrix_path)

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

    # save the models themselves for future inspection
    file_name = '{}.dill'.format(get_exp_id())
    print(type(question_models))
    save_path = dm.save_experiment(question_models, file_name=file_name)
    ex.add_artifact(save_path)
