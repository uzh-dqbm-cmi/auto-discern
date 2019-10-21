import os
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
import requests
import torch

import autodiscern.transformations as adt
from neural.data_processor import DataDictProcessor
from neural.dataset import generate_docpartition_per_question
from neural.model import BertEmbedder, generate_sents_embeds_from_docs
from neural.run_workflow import run_neural_discern, get_saved_config
from neural.utilities import create_directory, ReaderWriter
from neural.neural_discern_run_script import load_biobert_model
from typing import Dict, List


BASE_DIR = '/opt/data/autodiscern'
questions = [4, 5, 9, 10, 11]


def retrieve_predictions(dir):
    results = {}
    for q in questions:
        # path = os.path.join(dir, 'question_{}'.format(q), 'fold_0', 'predictions.csv')
        # results[q] = pd.read_csv(path)
        results[q] = 'not implemented'
    # TODO: are the predictions saved anywhere?
    return results


def run_predict(q_docpartitions, q_fold_config_map, bertmodel, q_state_dict_path_map, results_dir, sents_embed_dir,
                to_gpu, gpu_index, num_epochs=1):
    fold_num = 0
    dsettypes = ['test']
    q_predictions = {}
    for question in q_fold_config_map:
        mconfig, options, __ = q_fold_config_map[question]
        options['num_epochs'] = num_epochs  # override number of epochs using user specified value

        # update options fold num to the current fold
        options['fold_num'] = fold_num
        data_partition = q_docpartitions[question][fold_num]

        results_path = os.path.join(results_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))
        results_wrk_dir = create_directory(results_path)

        q_predictions[question] = run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options,
                                                     results_wrk_dir, sents_embed_dir,
                                                     state_dict_dir=q_state_dict_path_map[question], to_gpu=to_gpu,
                                                     gpu_index=gpu_index)
    return q_predictions


def retrieve_page_from_internet(url: str):
    res = requests.get(url)
    html_page = res.content.decode("utf-8")
    return html_page


def create_prediction_qdoc_partitions(questions: List[int], fold_num: int):
    q_docpartitions = {}
    for q in questions:
        q_docpartitions[q] = {}
        q_docpartitions[q][fold_num] = {'train': [], 'validation': [], 'test': [0]}  # output of np.vectorize() ?!
    return q_docpartitions


def build_docs_data_tensor(data_dict, vocab_path: str, processor_config: Dict):
    # from the first notebook

    processor = DataDictProcessor(processor_config)
    processor.generate_articles_repr(data_dict)

    tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)  # on LeoMed

    # generate docs data tensor from the articles i.e. instance of class DocDataTensor
    docs_data_tensor = processor.generate_doctensor_from_articles(tokenizer)
    return docs_data_tensor


def embed_sentences(docs_data_tensor, sents_embed_path, bertmodel, bert_config, gpu_index):
    # from the second notebook

    bertembeder = BertEmbedder(bertmodel, bert_config)
    fdtype = torch.float32

    # generate and dump bert embedding for the tokens inside the specificed embedding directory
    bert_proc_docs = generate_sents_embeds_from_docs(docs_data_tensor, bertembeder, sents_embed_path, fdtype,
                                                     gpu_index=gpu_index)
    ReaderWriter.dump_data(bert_proc_docs, os.path.join(sents_embed_path, 'bert_proc_docs.pkl'))

    return sents_embed_path


def biobert_predict(data_dict: dict):
    """
    Make a prediction for an article data_dict

    Args:
        data_dict: dictionary of {id: sib-dict}, with sub-dictionary with keys ['url', 'content', 'id']

    Returns: autodiscern predictions for the article

    """

    base_dir = BASE_DIR
    to_gpu = True
    gpu_index = 3
    fold_num = 0
    working_dir = 'predict'  # TODO
    experiment_dir = '2019-10-14_14-41-53'

    vocab_path = os.path.join(base_dir, 'aa_neural/aws_downloads/bert-base-cased-vocab.txt')
    # TODO: load config from filesystem?
    processor_config = {'tokenizer_max_sent_len': 300,
                        'label_cutoff': 3,
                        'label_avgmethod': 'round_mean'}

    sents_embed_dir = os.path.join(base_dir, 'aa_neural/sents_bert_embed_cased')  # TODO
    # TODO: load from filesystem?
    bert_config = {'bert_train_flag': False,
                   'bert_all_output': False}

    state_dict_path_form = 'train_validation/question_{}/fold_0/model_statedict/'  # TODO QUESTION:  why not in test?
    config_path_form = 'test/question_{}/fold_0/config/'

    # ---

    q_partitions = create_prediction_qdoc_partitions(questions, fold_num)

    # run data processing
    # USED "2019-05-02_15-49-09_a0745f9_sent_level_MM.pkl"
    html_to_sentence_transformer = adt.Transformer(leave_some_html=True,
                                                   html_to_plain_text=True,
                                                   segment_into='sentences',
                                                   flatten=True,
                                                   # remove_newlines=False,  # in newer version
                                                   annotate_html=True,
                                                   parallelism=False)
    transformed_data = html_to_sentence_transformer.apply(data_dict)

    # load BERT model
    pytorch_dump_path = os.path.join(base_dir, 'aa_neural', 'pytorch_biobert')
    bert_for_pretrain = load_biobert_model(pytorch_dump_path)
    bertmodel = bert_for_pretrain.bert

    docs_data_tensor = build_docs_data_tensor(transformed_data, vocab_path, processor_config)

    # create q_docpartitions
    q_docpartitions = {}
    for question in questions:
        q_docpartitions.update(generate_docpartition_per_question(docs_data_tensor, q_partitions, question))

    # embed sentences
    print("Embedding sentences...")
    sents_embed_dir = embed_sentences(docs_data_tensor, sents_embed_dir, bertmodel, bert_config, gpu_index)
    print(" ... Finished embedding sentences")

    # load model configs
    q_fold_config_map = {}
    for q in questions:
        config_path = os.path.join(base_dir, 'aa_neural', 'experiments', experiment_dir, config_path_form.format(q))
        mconfig, options = get_saved_config(config_path)
        argmax_indx = 'ignored'
        q_fold_config_map[q] = (mconfig, options, argmax_indx)

    # load model state_dicts
    q_state_dict_path_map = {}
    for q in questions:
        state_dict_path = os.path.join(base_dir, 'aa_neural', 'experiments', experiment_dir,
                                       state_dict_path_form.format(q))
        q_state_dict_path_map[q] = state_dict_path

    print("Running predict")
    results = run_predict(q_docpartitions, q_fold_config_map, bertmodel, q_state_dict_path_map, working_dir,
                          sents_embed_dir, to_gpu, gpu_index, num_epochs=1)
    return results


def build_data_dict(url, content):
    fake_responses = pd.DataFrame({'fake responses': [0]*12})  # is this supposed to be a len 5 or 11 list?
    # TODO: Q:question is used as index, not sure if that means list needs to be 11 long, or questions are actually 0-4
    data_dict = {0: {'id': 0,
                     'url': url,
                     'content': content,
                     'responses': fake_responses,
                     }
                 }
    return data_dict


def make_prediction(url: str):
    html_content = retrieve_page_from_internet(url)
    data_dict = build_data_dict(url, html_content)
    return biobert_predict(data_dict)


def test_make_prediction():
    test_data_path = os.path.join(BASE_DIR, 'data', 'test.html')
    test_article_url = 'https://www.nhs.uk/conditions/tendonitis/'

    with open(test_data_path, 'r') as f:
        html_content = f.read()
    data_dict = build_data_dict(test_article_url, html_content)
    return biobert_predict(data_dict)
