import os
import flask
import autodiscern.model as adm
from neural.predict_with_neural import make_prediction


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

    @app.route('/')
    def hello():
        return flask.render_template('index.html', url_str='', predictions={})

    @app.route('/validate', methods=['POST'])
    def validate():
        url = flask.request.form["url"]
        predictions = make_neural_prediction(url)
        return flask.render_template('index.html', url_str=url, predictions=predictions)

    @app.route('/test', methods=['POST'])
    def test():
        url = "www.dontgetsick.com"
        predictions = {
            4: {
                'question': "Is it clear what sources of information were used to compile the publication (other than "
                            "the author or producer)?",
                'answer': 'No',
                'sentences': [],
            },
            5: {
                'question': "Is it clear when the information used or reported in the publication was produced?",
                'answer': "No",
                'sentences': [],
            },
            9: {
                'question': "Does it describe how each treatment works?",
                'answer': "Yes",
                'sentences': ['The treatment works via the ephemeral properties of magic tacos.'],
            },
            10: {
                'question': "Does it describe the benefits of each treatment?",
                'answer': "Yes",
                'sentences': [
                    'The treatment ensures you never get sick.',
                    'The treatment ensures you never have to take a sick day.',
                ],
            },
            11: {
                'question': "Does it describe the risks of each treatment?",
                'answer': "No",
                'sentences': [],
            },
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
                predictions_to_display[q]['sentences'] = []
            else:
                predictions_to_display[q]['answer'] = 'Unknown'
                predictions_to_display[q]['sentences'] = []

        return predictions_to_display


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=80)
