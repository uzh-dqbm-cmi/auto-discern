import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Dict, List


def get_label_for_question(data_dict: Dict, question_no: int) -> float:
    r = data_dict['responses']
    return r.loc[question_no].mean()


def build_data_for_question_submodels(data: Dict, important_questions: List[int]) -> Dict:
    """

    Args:
        data: dict. A data dictionary {doc_id: {'content': "text", 'other_keys': 'values}, }
        important_questions: List[int]. List of questions to create datasets for.

    Returns:
        modeling_data = {
            submodel_id_question: {
                'X_train': [],
                'X_test': [],
                'y_train': [],
                'y_test': [],
            }
        }

    """
    # build dataset
    data_key_order = list(data.keys())
    corpus = [data[key]['content'] for key in data_key_order]

    # build labels
    labels = {q: [get_label_for_question(data[key], q) for key in data_key_order] for q in important_questions}
    labels['overall'] = [data[key]['responses'].median().median() for key in data_key_order]

    modeling_data = {}
    for submodel_id in list(labels.keys()):
        X_train, X_test, y_train, y_test = train_test_split(corpus, labels[submodel_id], test_size=0.33,
                                                            random_state=42)
        modeling_data[submodel_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }
    return modeling_data


def tfidf_data(data_dict, min_df=0.1, max_df=0.95):
    print("Initial data length: {:,.0f}".format(len(data_dict['X_train'])))
    data_dict['vectorizer'] = TfidfVectorizer(max_df=max_df, min_df=min_df)
    data_dict['X_train_tfidf'] = data_dict['vectorizer'].fit_transform(data_dict['X_train'])
    data_dict['X_test_tfidf'] = data_dict['vectorizer'].transform(data_dict['X_test'])
    print("X_train dims: {}".format(data_dict['X_train_tfidf'].shape))
    print("X_test dims: {}".format(data_dict['X_test_tfidf'].shape))
    return data_dict


def make_error_df(actual, predicted):
    results = pd.DataFrame({'actual': actual, 'predicted': predicted})
    results['perc_error'] = (results['actual'] - results['predicted']) / results['actual']
    return results


def density_scatter_plot(x, y, s=''):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('#06007A')
    plt.hist2d(x, y, (25, 25), cmap=plt.cm.jet)
    plt.xlim((1, 5))
    plt.ylim((1, 5))
    plt.colorbar()
    plt.title(s, fontsize=18, color='black')
    plt.show()


def compile_multiple_density_scatter_plots(data_dict):
    split_types = list(data_dict.keys())
    submodel_ids = list(data_dict[split_types[0]].keys())

    fig, ax = plt.subplots(len(split_types), len(submodel_ids), sharex='col', sharey='row', figsize=(16, 8))
    for i, split_type in enumerate(split_types):
        for j, submodel_id in enumerate(submodel_ids):
            model_data = data_dict[split_type][submodel_id]
            name = "{}, {}".format(split_type, submodel_id)
            if 'y_test_predict' in model_data.keys():
                mapes = "{:.0%}-{:.0%}".format(model_data['mape_train'], model_data['mape_test'])
                ax[i, j].hist2d(model_data['y_test'], model_data['y_test_predict'], (25, 25), cmap=plt.cm.jet)
                ax[i, j].set_title("{}, {}".format(name, mapes), fontsize=10, color='black')
                ax[i, j].set_xlim((1, 5))
                ax[i, j].set_ylim((1, 5))
                ax[i, j].set_facecolor('#06007A')
            else:
                ax[i, j].text(2.5, 2.5, name, fontsize=10, ha='center')
                ax[i, j].set_xlim((1, 5))
                ax[i, j].set_ylim((1, 5))

    plt.show()


def test_model(data: Dict, model):
    """Fit the model and calc error metrics"""

    model.fit(data['X_train_tfidf'], data['y_train'])
    data['model'] = model
    data['y_train_predict'] = model.predict(data['X_train_tfidf'])
    data['y_test_predict'] = model.predict(data['X_test_tfidf'])
    data['score'] = model.score(data['X_test_tfidf'], data['y_test'])

    results_train = make_error_df(data['y_train'], data['y_train_predict'])
    data['mape_train'] = 1 - abs(results_train['perc_error']).mean()

    results_test = make_error_df(data['y_test'], data['y_test_predict'])
    data['mape_test'] = 1 - abs(results_test['perc_error']).mean()
    return data
