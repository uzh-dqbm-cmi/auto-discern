import os
import flask

import autodiscern as ad
import autodiscern.transformations as adt
import autodiscern.annotations as ada
import autodiscern.model as adm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import inspect
import pickle
import requests


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

    model_filepath = os.path.join(package_dir, "data/models/2019_06_14_doc_models_important_qs.pkl")
    with open(model_filepath, "rb+") as f:
        ems = pickle.load(f)

    predictors = {}
    for q in ems:
        predictors[q] = ems[q].generate_predictor()

    @app.route('/')
    def hello():
        return flask.render_template('base.html', url_str='', predictions={})

    @app.route('/validate', methods=['POST'])
    def validate():
        url = flask.request.form["url"]
        predictions = make_prediction(predictors, url)
        return flask.render_template('base.html', url_str=url, predictions=predictions)

    def make_prediction(predictors, url: str):
        res = requests.get(url)
        html_page = res.content.decode("utf-8")
        data_dict = {0: {'entity_id': 0, 'content': html_page, 'url': url}}

        html_transformer = adt.Transformer(leave_some_html=True,
                                           html_to_plain_text=True,
                                           annotate_html=True,
                                           parallelism=False
                                           )
        transformed_data = html_transformer.apply(data_dict)
        transformed_data = ada.add_inline_citations_annotations(transformed_data)
        metamap_path = os.path.join(package_dir, "data/metamap_exec/public_mm_lite")
        transformed_data = ada.add_metamap_annotations(transformed_data, dm, metamap_path)

        sid = SentimentIntensityAnalyzer()

        for key in data_dict:
            transformed_data[key]['feature_vec'] = adm.build_remaining_feature_vector(transformed_data[key], sid)

        predictions = {}
        for q in predictors:
            question = "{}: {}".format(q, adm.questions[int(q.split('q')[1])])
            prediction = predictors[q].predict(transformed_data[0])
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
