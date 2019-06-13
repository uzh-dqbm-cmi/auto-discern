import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, List
from copy import deepcopy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer


questions = {
    1: "Are the aims clear?",
    2: "Does it achieve its aims?",
    3: "Is it relevant?",
    4: "Is it clear what sources of information were used to compile the publication (other than the author or "
       "producer)?",
    5: "Is it clear when the information used or reported in the publication was produced?",
    6: "Is it balanced and unbiased?",
    7: "Does it provide details of additional sources of support and information?",
    8: "Does it refer to areas of uncertainty?",
    9: "Does it describe how each treatment works?",
    10: "Does it describe the benefits of each treatment?",
    11: "Does it describe the risks of each treatment?",
    12: "Does it describe what would happen if no treatment is used?",
    13: "Does it describe how the treatment choices affect overall quality of life?",
    14: "Is it clear that there may be more than one possible treatment choice?",
    15: "Does it provide support for shared decision-making?",
}


def get_score_for_question(data_dict: Dict, question_no: int) -> float:
    r = data_dict['responses']
    return r.loc[question_no].mean()


def continuous_regression(score):
    return score


def neg_neutral_pos_category(score):
    if round(score) == 3:
        return 'neutral'
    elif round(score) < 3:
        return 'negative'
    elif round(score) > 3:
        return 'positive'
    else:
        raise(ValueError("No label applied"))


def neg_pos_category(score):
    if round(score) < 3:
        return 'negative'
    elif round(score) >= 3:
        return 'positive'
    else:
        raise(ValueError("No label applied"))


def zero_one_category(score):
    if round(score) < 3:
        return 0
    elif round(score) >= 3:
        return 1
    else:
        raise(ValueError("No label applied"))


def build_data_for_question_submodels(data: Dict, label_func: Callable = continuous_regression,
                                      important_questions: List[int] = [4, 5, 9, 10, 11]) -> Dict:
    """

    Args:
        data: dict. A data dictionary {doc_id: {'content': "text", 'other_keys': 'values}, }
        label_func: Function to use to generate the label from the score.
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
    import datetime
    start_timestamp = datetime.datetime.now()
    print("Starting function: {}".format(start_timestamp))

    # build dataset
    data_key_order = list(data.keys())
    sid = SentimentIntensityAnalyzer()
    corpus = [(data[key]['content'], build_remaining_feature_vector(data[key], sid)) for key in data_key_order]

    timestamp = datetime.datetime.now()
    print("Built corpus: {}".format(timestamp - start_timestamp))
    start_timestamp = timestamp

    # build labels
    labels = {q: [label_func(get_score_for_question(data[key], q)) for key in data_key_order]
              for q in important_questions}
    labels['overall'] = [label_func(data[key]['responses'].median().median()) for key in data_key_order]

    timestamp = datetime.datetime.now()
    print("Built labels: {}".format(timestamp - start_timestamp))
    start_timestamp = timestamp

    modeling_data = {}
    for submodel_id in list(labels.keys()):
        X_train, X_test, y_train, y_test = train_test_split(corpus, labels[submodel_id], test_size=0.33,
                                                            random_state=42)

        timestamp = datetime.datetime.now()
        print("Completed train test split: {}".format(timestamp - start_timestamp))
        start_timestamp = timestamp

        corpus_train = [t[0] for t in X_train]
        corpus_test = [t[0] for t in X_test]
        vec_train = pd.concat([t[1] for t in X_train], axis=0)
        vec_test = pd.concat([t[1] for t in X_test], axis=0)

        timestamp = datetime.datetime.now()
        print("Split out corpus and vec_train: {}".format(timestamp - start_timestamp))
        start_timestamp = timestamp

        modeling_data[submodel_id] = {
            'corpus_train': corpus_train,
            'corpus_test': corpus_test,
            'vec_train': vec_train,
            'vec_test': vec_test,
            'y_train': y_train,
            'y_test': y_test,
        }

        timestamp = datetime.datetime.now()
        print("Completed loop: {}".format(timestamp - start_timestamp))
        start_timestamp = timestamp

    return modeling_data


def tfidf_data(data_dict, min_df=0.1, max_df=0.95):
    print("Initial data length: {:,.0f}".format(len(data_dict['corpus_train'])))
    data_dict['vectorizer'] = TfidfVectorizer(max_df=max_df, min_df=min_df)
    data_dict['X_train_tfidf'] = data_dict['vectorizer'].fit_transform(data_dict['corpus_train'])
    data_dict['X_test_tfidf'] = data_dict['vectorizer'].transform(data_dict['corpus_test'])
    print("X_train_tfidf dims: {}".format(data_dict['X_train_tfidf'].shape))
    print("X_test_tfidf dims: {}".format(data_dict['X_test_tfidf'].shape))
    return data_dict


def build_remaining_feature_vector(data_dict, sid):
    return pd.concat([
        vectorize_html(data_dict['html_tags']),
        vectorize_link_type(data_dict['link_type']),
        vectorize_citations(data_dict['citations']),
        compute_polarity(data_dict['content'], sid),
        vectorize_metamap((data_dict.get('metamap', []))),
        compute_bibliography_feature(data_dict['content']),
        # TODO: dates
    ], axis=1)


def vectorize_html(input: List[str]) -> pd.DataFrame:
    return pd.DataFrame({
        'html_h1': 1 if 'h1' in input else 0,
        'html_h2': 1 if 'h2' in input else 0,
        'html_h3': 1 if 'h3' in input else 0,
        'html_h4': 1 if 'h4' in input else 0,
        'html_a': 1 if 'a' in input else 0,
        'html_li': 1 if 'li' in input else 0,
        'html_tr': 1 if 'tr' in input else 0,
    }, index=[0])


def vectorize_metamap(input: List[str]) -> pd.DataFrame:
    metamap_concept_groups = ['Chemicals & Drugs', 'Disorders', 'Activities & Behaviors', 'Living Beings',
                              'Genes & Molecular Sequences', 'Anatomy', 'Phenomena', 'Occupations', 'Physiology',
                              'Concepts & Ideas', 'Procedures', 'Devices', 'Objects', 'Geographic Areas',
                              'Organizations']
    metamap_vec = {}
    for group_name in metamap_concept_groups:
        metamap_vec["MM-{}".format(group_name)] = sum([1 for entry in input if entry == group_name])

    return pd.DataFrame(metamap_vec, index=[0])


def vectorize_link_type(input: List[str]) -> pd.DataFrame:
    return pd.DataFrame({
        'internal_link_cnt': sum([1 for link_type in input if link_type == 'internal']),
        'external_link_cnt': sum([1 for link_type in input if link_type == 'external']),
    }, index=[0])


def vectorize_citations(input: List[str]) -> pd.DataFrame:
    return pd.DataFrame({'inline_citation_cnt': len(input)}, index=[0])


def compute_polarity(input: str, sid: SentimentIntensityAnalyzer) -> pd.DataFrame:
    polarity_score_dict = sid.polarity_scores(input)
    polarity_score_dict = {"sentiment_{}".format(key): polarity_score_dict[key] for key in polarity_score_dict}
    return pd.DataFrame(polarity_score_dict, index=[0])


def compute_bibliography_feature(input_str: str) -> pd.DataFrame:
    bibliography_keywords = [
        'bibliography',
        'references',
        'works cited',
        'citations',
        'sources',
    ]
    if any(x in input_str.lower() for x in bibliography_keywords):
        value = 1
    else:
        value = 0
    return pd.DataFrame({'bibliography_feature': value}, index=[0])


def combine_features(data_dict):
    from scipy.sparse import hstack, coo_matrix

    print("X_train_tfidf dims: {}".format(data_dict['X_train_tfidf'].shape))
    print("vec_train dims: {}".format(data_dict['vec_train'].shape))
    data_dict['X_train'] = hstack([data_dict['X_train_tfidf'], coo_matrix(data_dict['vec_train'])])
    data_dict['X_test'] = hstack([data_dict['X_test_tfidf'], coo_matrix(data_dict['vec_test'])])
    data_dict['feature_cols'] = data_dict['vectorizer'].get_feature_names()
    data_dict['feature_cols'].extend(data_dict['vec_train'].columns)
    print("X_train dims: {}".format(data_dict['X_train'].shape))
    print("X_test dims: {}".format(data_dict['X_test'].shape))
    return data_dict


def make_error_df(actual, predicted):
    results = pd.DataFrame({'actual': actual, 'predicted': predicted})
    results['error'] = results['actual'] - results['predicted']
    results['abs_error'] = abs(results['error'])
    results['perc_error'] = (results['actual'] - results['predicted']) / results['actual']
    return results


def density_scatter_plot(x, y, s=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('#06007A')
    plt.hist2d(x, y, (25, 25), cmap=plt.cm.jet)
    plt.xlim((1, 5))
    plt.ylim((1, 5))
    plt.colorbar()
    plt.title(s, fontsize=18, color='black')
    plt.show()


def test_model(data: Dict, model, type='regression'):
    """Fit the model and calc error metrics"""

    model.fit(data['X_train'], data['y_train'])
    data['model'] = model
    data['y_train_predict'] = model.predict(data['X_train'])
    data['y_test_predict'] = model.predict(data['X_test'])
    data['score'] = model.score(data['X_test'], data['y_test'])

    if type == 'regression':
        results_train = make_error_df(data['y_train'], data['y_train_predict'])
        data['mae_train'] = results_train['abs_error'].mean()
        data['mape_train'] = 1 - abs(results_train['perc_error']).mean()

        results_test = make_error_df(data['y_test'], data['y_test_predict'])
        data['mae_test'] = results_test['abs_error'].mean()
        data['mape_test'] = 1 - abs(results_test['perc_error']).mean()
    elif type == 'classification':
        data['confusion_matrix'] = confusion_matrix(data['y_test'], data['y_test_predict'])
    return data


def run_random_cv(data_dict, n_iter=5, cv=3):
    # Number of trees in random forest
    n_estimators = [100, 500]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [10, 100, None]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 50, 500]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 10, 100]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   # 'bootstrap': bootstrap,
                   'class_weight': ['balanced_subsample'],
                   }
    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=n_iter, cv=cv, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(data_dict['X_train'], data_dict['y_train'])
    return rf_random


def compare_base_to_random_cv(data_dict, n_iter=5, cv=3):
    base_model = RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample', random_state=42)
    data_dict_base = test_model(deepcopy(data_dict), base_model, type='classification')

    rf_random = run_random_cv(deepcopy(data_dict), n_iter=n_iter, cv=cv)
    print(rf_random.best_params_)
    best_random_model = rf_random.best_estimator_
    data_dict_random = test_model(data_dict, best_random_model, type='classification')

    print("Base accuracy:      {:.2%}".format(data_dict_base['score']))
    print("Random CV accuracy: {:.2%}".format(data_dict_random['score']))
    print('Improvement of {:.2%}.'.format(data_dict_random['score'] - data_dict_base['score']))
    return data_dict_base, data_dict_random


def compile_multiple_density_scatter_plots(data_dict):
    import matplotlib.pyplot as plt

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


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
    cm_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm_p, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm_p.shape[1]),
           yticks=np.arange(cm_p.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm_p.max() / 2.
    for i in range(cm_p.shape[0]):
        for j in range(cm_p.shape[1]):
            ax.text(j, i, "{:.0%}\n{}".format(cm_p[i, j], cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm_p[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def tile_images_into_square(image_list):
    """Take a list of image file paths, and tile all
    of the images into one in a square pattern."""
    from PIL import Image
    from math import ceil, sqrt

    images = map(Image.open, image_list)
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)
    num_images = len(widths)
    img_per_side = int(ceil(sqrt(num_images)))
    total_width = img_per_side * max_width
    total_height = img_per_side * max_height

    new_img = Image.new('RGB', (total_width, total_height))

    images = map(Image.open, image_list)
    for i, im in enumerate(images):
        x_pos = (i % img_per_side) * max_width
        y_pos = (int(i / img_per_side)) * max_height
        new_img.paste(im, (x_pos, y_pos))

    return new_img


def tile_images_into_rectangle_of_width(image_list, width=3):
    """Take a list of image file paths, and tile all
    of the images into one in a square pattern."""
    from PIL import Image
    from math import ceil, floor

    images = map(Image.open, image_list)
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)
    num_images = len(widths)
    img_per_height = int(ceil(num_images/width))
    img_per_width = width
    total_width = img_per_width * max_width
    total_height = img_per_height * max_height

    new_img = Image.new('RGB', (total_width, total_height))

    images = map(Image.open, image_list)
    for i, im in enumerate(images):
        x_pos = (i % img_per_width) * max_width
        y_pos = int(floor(i / img_per_width) * max_height)
        new_img.paste(im, (x_pos, y_pos))

    return new_img


def compile_multiple_confusion_matrices(data_dict):
    import matplotlib.pyplot as plt

    split_types = [key for key in ['doc', 'para', 'sent'] if key in data_dict.keys()]
    submodel_ids = list(data_dict[split_types[0]].keys())

    image_filenames = []

    # fig, ax = plt.subplots(len(split_types), len(submodel_ids), figsize=(16, 8))
    for j, submodel_id in enumerate(submodel_ids):
        for i, split_type in enumerate(split_types):
            model_data = data_dict[split_type][submodel_id]
            name = "{}, {}".format(split_type, submodel_id)
            if 'y_test_predict' in model_data.keys():
                score = "{:.2}".format(model_data['score'])
                # ax[i, j] = \
                # apparently you don't need to save the returned image?!
                plot_confusion_matrix(model_data['y_test'], model_data['y_test_predict'],
                                      classes=model_data['model'].classes_, normalize=True,
                                      title='{}, {}'.format(name, score))
                filename = 'images/{}.png'.format(name)
                image_filenames.append(filename)
                plt.savefig(filename)
    #         else:
    #             ax[i, j].text(2.5, 2.5, name, fontsize=10, ha='center')
    #             ax[i, j].set_xlim((1, 5))
    #             ax[i, j].set_ylim((1, 5))
    #
    # plt.show()
    return tile_images_into_rectangle_of_width(image_filenames, 3)


def get_feature_importances(data_dict: Dict) -> pd.DataFrame:
    fi = pd.DataFrame(data_dict['model'].feature_importances_, index=data_dict['feature_cols'])
    fi = fi.sort_values(0, ascending=False)
    return fi


def show_top_features_importances(modeling_data: Dict, n=10) -> None:
    """Print the top n features for each model, organized by question and then split type"""
    top_level_keys = list(modeling_data.keys())
    for submodel_id in modeling_data[top_level_keys[0]]:
        print("q{}: {}".format(submodel_id, questions[submodel_id]))
        for split_type in modeling_data:
            data_dict = modeling_data[split_type][submodel_id]
            fi = get_feature_importances(data_dict)
            print('  {}: {}'.format(split_type, ", ".join(list(fi.head(n).index))))
        print()


def calc_base_rate_accuracy(modeling_data: Dict) -> pd.DataFrame:
    """Calculate the Base Rate accuracy (if we made a model to always choose majority class). """
    scores = []
    for split_type in modeling_data:
        for submodel_id in modeling_data[split_type]:
            labels = modeling_data[split_type][submodel_id]['y_test']
            base_rate_neg = sum([1 for lab in labels if lab == 'negative']) / len(labels)
            base_rate_pos = sum([1 for lab in labels if lab == 'positive']) / len(labels)
            if base_rate_neg > base_rate_pos:
                base_rate = base_rate_neg
            else:
                base_rate = base_rate_pos
            scores.append({'split': split_type,
                           'question': submodel_id,
                           'base_rate': base_rate})
    scores_df = pd.DataFrame(scores)
    base_rate_pivot = pd.pivot_table(scores_df, index='split', columns='question', values='base_rate', aggfunc='median')
    return base_rate_pivot


def calc_accuracy(modeling_data: Dict) -> pd.DataFrame:
    scores = []
    for split_type in modeling_data:
        for submodel_id in modeling_data[split_type]:
            scores.append({'split': split_type, 'question': submodel_id,
                           'score': modeling_data[split_type][submodel_id]['score']})
    scores_df = pd.DataFrame(scores)
    accuracy_pivot = pd.pivot_table(scores_df, index='split', columns='question', values='score', aggfunc='median')
    return accuracy_pivot


def calc_in_sample_accuracy(modeling_data: Dict) -> pd.DataFrame:
    """Calculate the in-sample score on the training sets and return a dataframe by question and split type"""
    in_sample_scores = []
    for split_type in modeling_data:
        for submodel_id in modeling_data[split_type]:
            d = modeling_data[split_type][submodel_id]
            in_sample_score = d['model'].score(d['X_train'], d['y_train'])

            in_sample_scores.append({
                'split': split_type,
                'question': submodel_id,
                'in_sample_score': in_sample_score
            })
    in_sample_scores_df = pd.DataFrame(in_sample_scores)
    pivot = pd.pivot_table(in_sample_scores_df, index='split', columns='question', values='in_sample_score',
                           aggfunc='median')
    return pivot
