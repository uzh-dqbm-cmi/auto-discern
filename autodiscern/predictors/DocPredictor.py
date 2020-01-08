import autodiscern as ad
import autodiscern.transformations as adt
import autodiscern.annotations as ada
import autodiscern.model as adm
from autodiscern.predictor import Predictor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import requests
from typing import Dict


discern_data_path = "~/switchdrive/Institution/discern"
dm = ad.DataManager(discern_data_path)

important_qs = [4, 5, 9, 10, 11]

filepath = "/Users/laurakinkead/switchdrive/Institution/discern/models/2019_06_14_doc_models_important_qs.pkl"
with open(filepath, "rb+") as f:
    ems = pickle.load(f)

predictors = {}
for q in ems:
    predictors[q] = ems[q].generate_predictor()


def make_prediction(predictors: Dict[Predictor], url: str) -> Dict:
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
    transformed_data = ada.add_metamap_annotations(transformed_data, dm)

    sid = SentimentIntensityAnalyzer()

    for key in data_dict:
        transformed_data[key]['feature_vec'] = adm.build_remaining_feature_vector(transformed_data[key], sid)

    predictions = {}
    for q in predictors:
        predictions[q] = predictors[q].predict(data_dict[0])

    return predictions
