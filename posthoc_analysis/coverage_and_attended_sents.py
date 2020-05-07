import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, accuracy_score
from neural.run_workflow import return_attnw_over_sents
from neural.utilities import ReaderWriter


precision_threshold_limit = 0.90

questions = [4, 5, 9, 10, 11]
folds = [0, 1, 2, 3, 4]

# experiment = '2019-10-08_14-54-50_rerun_2019-11-14_09-38-59'  # bert
experiment = '2019-10-28_15-59-09_rerun_2019-11-14_14-38-53'  # biobert

all_predictions = pd.DataFrame()
results = pd.DataFrame()
coverage_results = pd.DataFrame()

for q in questions:
    for f in folds:
        pred_path = '/opt/data/autodiscern/aa_neural/experiments/{}/test/question_{}/fold_{}/predictions.csv'.format(
            experiment, q, f)

        if not os.path.exists(pred_path):
            continue

        pred = pd.read_csv(pred_path)
        pred['experiment'] = experiment
        pred['question'] = q
        pred['fold'] = f
        if 'logprob_score_class0' not in pred:
            pred['logprob_score_class0'] = pred['logprob_scores_class0']
            pred['logprob_score_class1'] = pred['logprob_scores_class1']

        pred['prob_score_class0'] = np.exp(pred['logprob_score_class0'])
        pred['prob_score_class1'] = np.exp(pred['logprob_score_class1'])
        pred['prob_check'] = pred['prob_score_class0'] + pred['prob_score_class1']
        all_predictions = pd.concat([all_predictions, pred])

        y_true = pred['true_class']
        y_scores = pred['prob_score_class1']
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_df = pd.DataFrame({'precision': precision, 'recall': recall, 'threshold': np.append(thresholds, 1)})

        # find the precision value that is closest but not exceding precision_threshold_limit
        precision_threshold = pr_df[pr_df['precision'] <= precision_threshold_limit]['precision'].max()
        fold_result = pr_df[pr_df['precision'] >= precision_threshold].head(1)

        fold_result['experiment'] = experiment
        fold_result['question'] = q
        fold_result['fold'] = f
        results = pd.concat([results, fold_result])

        pred['correct'] = pred['true_class'] == pred['pred_class']
        pred['confidence'] = pred[['prob_score_class0', 'prob_score_class1']].max(axis=1)

        for coverage_percent in [0.67, 0.75, 0.8, 0.9, 1]:
            coverage_threshold = pred['confidence'].quantile(1 - coverage_percent)
            covered = pred[(pred['confidence'] >= coverage_threshold)].copy()
            diff = abs(covered.shape[0] - pred.shape[0] * coverage_percent)
            assert diff < 2
            y_true = covered['true_class']
            y_pred = covered['pred_class']
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            fold_coverage_df = pd.DataFrame({
                'question': [q],
                'fold': [f],
                'coverage': [coverage_percent],
                'precision': [precision],
                'recall': [recall],
                'accuracy': [accuracy],
                'threshold': [coverage_threshold],
            })
            coverage_results = pd.concat([coverage_results, fold_coverage_df])

            if q == 10 and f == 2:
                confusion_matrix = pd.pivot_table(covered, index='true_class', columns='pred_class', values='id',
                                                  aggfunc='count')
                for i in [0, 1]:
                    if i not in confusion_matrix.columns:
                        confusion_matrix[i] = 0
                # print('-' * 100)
                # print(fold_coverage_df)
                # print(confusion_matrix.fillna(0)[[0, 1]])
                # print()

# print(results[['question', 'fold', 'precision', 'recall', 'threshold']])
# print(results.groupby('question').agg({'recall': 'mean'}))
coverage_summary = coverage_results.groupby(['question', 'coverage']
                                            ).agg({'precision': 'mean',
                                                   'recall': 'mean',
                                                   'accuracy': 'mean',
                                                   'threshold': 'mean',
                                                   })
coverage_summary

# === ATTENDED SENTENCES ===
# show top sentence for each question for the top 3 documents
#   that were predicted as 1, both for true 1 and true 0

data_dir = '/opt/data/autodiscern/aa_neural/proc_data_uncased'
proc_articles_repr = ReaderWriter.read_data(os.path.join(data_dir, 'processor_articles_repr.pkl'))

test_dir = '/opt/data/autodiscern/aa_neural/experiments/{}/test'.format(experiment)

for q in questions:
    print("\nQuestion {}".format(q))
    print(" Correctly predicted class 1")
    docs_df = all_predictions[(all_predictions['question'] == q) &
                              (all_predictions['true_class'] == 1)
                              ].sort_values('prob_score_class1', ascending=False).head(3)
    # print(docs_df)
    doc_ids = list(docs_df['id'])
    # print(doc_ids)

    for i, row in docs_df.iterrows():
        f = row['fold']
        docid_attnw_map_val = ReaderWriter.read_data(
            os.path.join(test_dir, 'question_{}/fold_{}/docid_attnw_map_test.pkl'.format(q, f)))
        attended_sents = return_attnw_over_sents(docid_attnw_map_val, proc_articles_repr, topk=1)
        doc_id = row['id']
        sentence = attended_sents[doc_id][0]['sentence']
        print("    Doc id {}: {}".format(row['id'], sentence))

    # ===
    print(" Incorrectly predicted class 1")
    docs_df = all_predictions[(all_predictions['question'] == q) &
                              (all_predictions['true_class'] == 0)
                              ].sort_values('prob_score_class1', ascending=False).head(3)
    # print(docs_df)
    doc_ids = list(docs_df['id'])
    # print(doc_ids)

    for i, row in docs_df.iterrows():
        f = row['fold']
        docid_attnw_map_val = ReaderWriter.read_data(
            os.path.join(test_dir, 'question_{}/fold_{}/docid_attnw_map_test.pkl'.format(q, f)))
        attended_sents = return_attnw_over_sents(docid_attnw_map_val, proc_articles_repr, topk=1)
        doc_id = row['id']
        sentence = attended_sents[doc_id][0]['sentence']
        print("    Doc id {}: {}".format(row['id'], sentence))


# === get one positive example per disease category
target_ids = pd.read_csv('/opt/data/autodiscern/data/target_ids.csv')
all_predictions = pd.merge(all_predictions, target_ids, left_on='id', right_on='entity_id')

disease_category_map = {
    2: 'breast cancer',
    4: 'arthritis',
    5: 'depression',
}

data_dir = '/opt/data/autodiscern/aa_neural/proc_data_uncased'
proc_articles_repr = ReaderWriter.read_data(os.path.join(data_dir, 'processor_articles_repr.pkl'))

test_dir = '/opt/data/autodiscern/aa_neural/experiments/{}/test'.format(experiment)

for q in questions:
    print("\nQuestion {}".format(q))
    for disease_id in disease_category_map:
        print(" Disease Category {}".format(disease_category_map[disease_id]))
        docs_df = all_predictions[(all_predictions['question'] == q) &
                                  (all_predictions['categoryName'] == disease_id) &
                                  (all_predictions['true_class'] == 1)
                                  ].sort_values('prob_score_class1', ascending=False).head(3)
        # print(docs_df)
        doc_ids = list(docs_df['id'])
        # print(doc_ids)

        for i, row in docs_df.iterrows():
            f = row['fold']
            docid_attnw_map_val = ReaderWriter.read_data(
                os.path.join(test_dir, 'question_{}/fold_{}/docid_attnw_map_test.pkl'.format(q, f)))
            attended_sents = return_attnw_over_sents(docid_attnw_map_val, proc_articles_repr, topk=1)
            doc_id = row['id']
            sentence = attended_sents[doc_id][0]['sentence']
            print("    Doc id {}: {}".format(row['id'], sentence))
