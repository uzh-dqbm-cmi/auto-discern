import os
import flask

import autodiscern as ad
import autodiscern.transformations as adt
import autodiscern.annotations as ada
import autodiscern.model as adm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from neural.predict_with_neural import make_prediction
import inspect
import pickle
import requests

NEURAL = True

# Traditional RF Model Configs
RF_MODEL_PREDICTOR_FILE_PATH = 'data/models/2019_06_14_doc_models_important_qs.pkl'

# Neural (BioBERT) Model Configs
DEFAULT_NEURAL_EXP_DIR = '2019-10-28_15-59-09'
DEFAULT_USE_GPU = False
DEFAULT_QUESTION_FOLD_MAP = {
    4: 0,
    5: 0,
    9: 0,
    10: 0,
    11: 0,
}


def create_app(test_config=None):
    # create and configure the app
    app = flask.Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # set up all model infrastructure
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    # the package directory is the grandparent of this file
    package_dir = os.path.abspath(os.path.dirname(os.path.dirname(this_file_path)))
    dm = ad.DataManager(package_dir)

    predictor_filepath = os.path.join(package_dir, RF_MODEL_PREDICTOR_FILE_PATH)
    with open(predictor_filepath, "rb+") as f:
        ems = pickle.load(f)

    predictors = {}
    for q in ems:
        predictors[q] = ems[q].generate_predictor()

    @app.route('/')
    def hello():
        return flask.render_template('index.html', url_str='', predictions={})

    @app.route('/validate', methods=['POST'])
    def validate():
        url = flask.request.form["url"]
        if NEURAL:
            predictions = make_neural_prediction(url)
        else:
            predictions = make_rf_prediction(predictors, url)
        return flask.render_template('index.html', url_str=url, predictions=predictions)

    @app.route('/test', methods=['POST'])
    def test():
        # test = flask.request.form["test"]
        url = "www.dontgetsick.com"
        predictions = {
            "q4: Is it clear what sources of information were used to compile the publication (other than the "
            "author or producer)?": "No",
            "q5: Is it clear when the information used or reported in the publication was produced?": "No",
            "q9: Does it describe how each treatment works?": "Yes",
            "q10: Does it describe the benefits of each treatment?": "Yes",
            "q11: Does it describe the risks of each treatment?": "No",
        }
        return flask.render_template('index.html', url_str=url, predictions=predictions)

    def make_neural_prediction(url: str, exp_dir=DEFAULT_NEURAL_EXP_DIR, question_fold_map=None, to_gpu=DEFAULT_USE_GPU,
                               gpu_index=0):
        predictions = make_prediction(url, exp_dir=exp_dir, question_fold_map=question_fold_map, to_gpu=to_gpu,
                                      gpu_index=gpu_index)

        predictions_to_display = {}
        for q in predictions:
            predictions_to_display[q] = {}
            predictions_to_display[q]['question'] = "Q{}: {}".format(q, adm.questions[q])
            if predictions[q]['pred_class'] == 1:
                predictions_to_display[q]['answer'] = 'Yes'
                predictions_to_display[q]['sentences'] = predictions[q]['sentences']
            elif predictions[q]['pred_class'] == 0:
                predictions_to_display[q]['answer'] = 'No'
            else:
                predictions_to_display[q]['answer'] = 'Unknown'

        return predictions_to_display

    def make_rf_prediction(predictors, url: str):
        # load the webpage content
        res = requests.get(url)
        html_page = res.content.decode("utf-8")
        data_dict = {'content': html_page, 'url': url}

        # run data processing
        # TODO: refactor so that this logic is tied to model directly
        html_transformer = adt.Transformer(leave_some_html=True,
                                           html_to_plain_text=True,
                                           annotate_html=True,
                                           parallelism=False
                                           )
        transformed_data = html_transformer.apply(data_dict)
        transformed_data = ada.add_inline_citations_annotations(transformed_data)
        # TODO: refactor metamap path to be in config or something
        metamap_path = os.path.join(package_dir, "data/metamap_exec/public_mm_lite")
        transformed_data = ada.add_metamap_annotations(transformed_data, dm, metamap_path)

        sid = SentimentIntensityAnalyzer()

        for key in data_dict:
            transformed_data[key]['feature_vec'] = adm.build_remaining_feature_vector(transformed_data[key], sid)

        predictions = {}
        for q in predictors:
            prediction = predictors[q].predict(transformed_data[0])

            question = "{}: {}".format(q, adm.questions[int(q.split('q')[1])])
            if prediction[0] == 0:
                predictions[question] = 'No'
            elif prediction[0] == 1:
                predictions[question] = 'Yes'
            else:
                predictions[question] = 'Unknown'

        return predictions

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=80)
