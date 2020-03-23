import os
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
import requests
import torch

import autodiscern.transformations as adt
from neural.data_processor import DataDictProcessor
from neural.dataset import generate_docpartition_per_question
from neural.model import BertEmbedder, generate_sents_embeds_from_docs
from neural.run_workflow import predict_neural_discern, get_saved_config, return_attnw_over_sents
from neural.utilities import create_directory, ReaderWriter, get_device
from neural.neural_discern_run_script import load_biobert_model
from typing import Dict, List


QUESTIONS = [4, 5, 9, 10, 11]
DEFAULT_BASE_DIR = '/opt/data/autodiscern'
DEFAULT_BIOBERT_EXP_DIR = '2019-10-28_15-59-09'
DEFAULT_QUESTION_FOLD_MAP = {
    4: 0,
    5: 0,
    9: 0,
    10: 0,
    11: 0,
}


def identify_attended_senteces(docid_attnweights_map, proc_articles_repr, topk=5):
    attended_sents = return_attnw_over_sents(docid_attnweights_map, proc_articles_repr, topk)
    return attended_sents


def run_predict(q_docpartitions, q_fold_config_map, bertmodel, q_state_dict_path_map, results_dir, sents_embed_dir,
                question_fold_map, to_gpu, gpu_index, num_epochs=1):
    q_predictions = {}
    for question in q_fold_config_map:
        mconfig, options, __ = q_fold_config_map[question]
        options['num_epochs'] = num_epochs  # override number of epochs using user specified value

        # update options fold num to the current fold
        options['fold_num'] = question_fold_map[question]
        data_partition = q_docpartitions[question][options['fold_num']]

        results_path = os.path.join(results_dir, 'question_{}'.format(question), 'fold_{}'.format(options['fold_num']))
        results_wrk_dir = create_directory(results_path)

        q_predictions[question] = predict_neural_discern(data_partition, bertmodel, mconfig, options,
                                                         results_wrk_dir, sents_embed_dir,
                                                         state_dict_dir=q_state_dict_path_map[question], to_gpu=to_gpu,
                                                         gpu_index=gpu_index)
    return q_predictions


def retrieve_page_from_internet(url: str):
    res = requests.get(url)
    html_page = res.content.decode("utf-8")
    return html_page


def create_prediction_qdoc_partitions(questions: List[int], question_fold_map: Dict[int, int]):
    q_docpartitions = {}
    for q in questions:
        q_docpartitions[q] = {}
        fold_num = question_fold_map[q]
        q_docpartitions[q][fold_num] = {'test': [0]}  # output of np.vectorize() ?!
    return q_docpartitions


def build_DataDictProcessor(data_dict, vocab_path: str, processor_config: Dict):
    # from the first notebook

    processor = DataDictProcessor(processor_config)
    processor.generate_articles_repr(data_dict)
    return processor


def embed_sentences(docs_data_tensor, sents_embed_path, bertmodel, bert_config, gpu_index):
    # from the second notebook

    bertembeder = BertEmbedder(bertmodel, bert_config)
    fdtype = torch.float32

    # generate and dump bert embedding for the tokens inside the specificed embedding directory
    bert_proc_docs = generate_sents_embeds_from_docs(docs_data_tensor, bertembeder, sents_embed_path, fdtype,
                                                     gpu_index=gpu_index)
    ReaderWriter.dump_data(bert_proc_docs, os.path.join(sents_embed_path, 'bert_proc_docs.pkl'))


def biobert_predict(data_dict: dict, questions, experiment_dir, base_dir, question_fold_map, to_gpu, gpu_index):
    """
    Make an autoDiscern prediction for an article data_dict using the HEA BioBERT model. Includes all of the data
    preprocessing steps as were applied for the training of the HEA BioBERT model.

    Args:
        data_dict: dictionary of {id: sib-dict}, with sub-dictionary with keys ['url', 'content', 'id', 'responses']

    Returns: autodiscern predictions for the article.

    """

    working_dir = 'predict'

    vocab_path = os.path.join(base_dir, 'aa_neural/aws_downloads/bert-base-cased-vocab.txt')
    processor_config = {'tokenizer_max_sent_len': 300,
                        'label_cutoff': 3,
                        'label_avgmethod': 'round_mean'}

    sents_embed_dir = os.path.join(base_dir, 'aa_neural/sents_bert_embed_cased')
    bert_config = {'bert_train_flag': False,
                   'bert_all_output': False}

    state_dict_path_form = 'train_validation/question_{}/fold_{}/model_statedict/'
    config_path_form = 'test/question_{}/fold_0/config/'

    default_device = get_device(to_gpu=False)

    # ---

    q_partitions = create_prediction_qdoc_partitions(questions, question_fold_map)

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
    bert_for_pretrain = load_biobert_model(pytorch_dump_path, default_device)
    bertmodel = bert_for_pretrain.bert

    processor = build_DataDictProcessor(transformed_data, vocab_path, processor_config)
    tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)

    # generate docs data tensor from the articles i.e. instance of class DocDataTensor
    docs_data_tensor = processor.generate_doctensor_from_articles(tokenizer)

    # create q_docpartitions
    q_docpartitions = {}
    for question in questions:
        q_docpartitions.update(generate_docpartition_per_question(docs_data_tensor, q_partitions, question))

    # embed sentences
    print("Embedding sentences...")
    embed_sentences(docs_data_tensor, sents_embed_dir, bertmodel, bert_config, gpu_index)
    print(" ... Finished embedding sentences")

    # load model configs
    q_fold_config_map = {}
    for q in questions:
        config_path = os.path.join(base_dir, 'aa_neural', 'experiments', experiment_dir, config_path_form.format(q))
        mconfig, options = get_saved_config(config_path)
        argmax_indx = -1
        q_fold_config_map[q] = (mconfig, options, argmax_indx)

    # load model state_dicts
    q_state_dict_path_map = {}
    for q in questions:
        state_dict_path = os.path.join(base_dir, 'aa_neural', 'experiments', experiment_dir,
                                       state_dict_path_form.format(q, question_fold_map[q]))
        q_state_dict_path_map[q] = state_dict_path

    print("Running predict")
    results = run_predict(q_docpartitions, q_fold_config_map, bertmodel, q_state_dict_path_map, working_dir,
                          sents_embed_dir, question_fold_map, to_gpu, gpu_index, num_epochs=1)

    proc_articles_repr = processor.articles_repr
    # TODO: do not run this if monfig['attnmodel_config'] is empty dict
    # currently if model ran with no attention then 'attention_weight_map' will be {} (i.e. empty dict)
    for q in results:
        results[q]['attended_sentences'] = identify_attended_senteces(results[q]['attention_weight_map'],
                                                                      proc_articles_repr)
    return results


def build_data_dict(url, content):
    """Format the html information in a data_dict as required for the prediction routine.

    Args:
        url: url of the article
        content: html contents of the article

    Returns: Dict
    """
    fake_responses = pd.DataFrame({'fake responses': [0]*5})
    data_dict = {0: {'id': 0,
                     'url': url,
                     'content': content,
                     'responses': fake_responses,
                     }
                 }
    return data_dict


def make_prediction(url: str, exp_dir=DEFAULT_BIOBERT_EXP_DIR, base_dir=DEFAULT_BASE_DIR, question_fold_map=None,
                    to_gpu=True, gpu_index=0):
    """
    End to end function for making an autoDiscern prediction for a given url.

    Args:
        url: url of the article to make predictions for
        exp_dir: experiment directory from which to retrieve the trained model.
        base_dir: the path to the base directory (up to and including including `autodiscern/aa_neural/`.
        question_fold_map: Dictionary mapping question number to fold number to use for the model.
        to_gpu: whether to run on GPU.
        gpu_index: index of gpu to use.

    Returns: autoDiscern predictions for the article

    """
    if question_fold_map is None:
        question_fold_map = DEFAULT_QUESTION_FOLD_MAP

    if not to_gpu:
        print("WARNING: to_gpu=False is not supported yet")

    html_content = retrieve_page_from_internet(url)
    data_dict = build_data_dict(url, html_content)
    return biobert_predict(data_dict, QUESTIONS, exp_dir, base_dir, question_fold_map, to_gpu, gpu_index)


def test_make_prediction(exp_dir=DEFAULT_BIOBERT_EXP_DIR, base_dir=DEFAULT_BASE_DIR, question_fold_map=None,
                         to_gpu=True, gpu_index=0):
    """
    End to end test function for making an autoDiscern prediction, without relying on an internet connection.
    Relies on a the existence of a test.html file.

    Returns: autoDiscern predictions for the article

    """

    if question_fold_map is None:
        question_fold_map = DEFAULT_QUESTION_FOLD_MAP

    if not to_gpu:
        print("WARNING: to_gpu=False is not supported yet")

    test_data_path = os.path.join(DEFAULT_BASE_DIR, 'data', 'test.html')
    test_article_url = 'https://www.nhs.uk/conditions/tendonitis/'

    with open(test_data_path, 'r') as f:
        html_content = f.read()
    data_dict = build_data_dict(test_article_url, html_content)
    return biobert_predict(data_dict, QUESTIONS, exp_dir, base_dir, question_fold_map, to_gpu, gpu_index)
