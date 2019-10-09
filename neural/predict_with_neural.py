import os
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


BASE_DIR = '/somewhere/goodness/knows/where'


def retrieve_predictions(dir):
    # TODO: are the predictions saved anywhere?
    return {}


def run_predict(q_docpartitions, q_fold_config_map, bertmodel, state_dict_path, test_dir, sents_embed_dir, to_gpu,
                gpu_index, num_epochs=1):
    fold_num = 0
    dsettypes = ['test']
    for question in q_fold_config_map:
        mconfig, options, __ = q_fold_config_map[question]
        options['num_epochs'] = num_epochs  # override number of epochs using user specified value

        # update options fold num to the current fold
        options['fold_num'] = fold_num
        data_partition = q_docpartitions[question][fold_num]

        path = os.path.join(test_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))
        test_wrk_dir = create_directory(path)

        run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options, test_wrk_dir, sents_embed_dir,
                           state_dict_dir=state_dict_path, to_gpu=to_gpu, gpu_index=gpu_index)
        return retrieve_predictions(test_wrk_dir)


def retrieve_page_from_internet(url: str):
    res = requests.get(url)
    html_page = res.content.decode("utf-8")
    return html_page


def create_prediction_qdoc_partitions(questions: List[int], fold_num: int):
    q_docpartitions = {}
    for q in questions:
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


def embed_sentences(docs_data_tensor, sents_embed_path, bertmodel, bert_config):
    # from the second notebook

    bertembeder = BertEmbedder(bertmodel, bert_config)
    fdtype = torch.float32

    # generate and dump bert embedding for the tokens inside the specificed embedding directory
    bert_proc_docs = generate_sents_embeds_from_docs(docs_data_tensor, bertembeder, sents_embed_path, fdtype)
    ReaderWriter.dump_data(bert_proc_docs, os.path.join(sents_embed_path, 'bert_proc_docs.pkl'))

    return sents_embed_path


def predict(url: str):
    base_dir = BASE_DIR
    to_gpu = False
    gpu_index = 0
    questions = [4, 5, 9, 10, 11]
    fold_num = 0
    working_dir = ''  # TODO

    vocab_path = '/opt/data/autodiscern/aa_neural/aws_downloads/bert-base-uncased-vocab.txt'
    # TODO: load config from filesystem?
    processor_config = {'tokenizer_max_sent_len': 300,
                        'label_cutoff': 3,
                        'label_avgmethod': 'round_mean'}

    sents_embed_dir = ''  # TODO
    # TODO: load from filesystem?
    bert_config = {'bert_train_flag': False,
                   'bert_all_output': False}

    state_dict_path = ''  # TODO a la 'train_validation/question_{}/fold_{}/model_state_dict/'
    config_path = '{}'  # TODO a la 'train_validation/question_{}/fold_{}/config/'

    # ---

    html_page = retrieve_page_from_internet(url)
    data_dict = {'content': html_page, 'url': url, 'id': 0}
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
    pytorch_dump_path = create_directory('pytorch_biobert', base_dir)
    bert_for_pretrain = load_biobert_model(pytorch_dump_path)
    bertmodel = bert_for_pretrain.bert

    docs_data_tensor = build_docs_data_tensor(transformed_data, vocab_path, processor_config)

    # create q_docpartitions
    q_docpartitions = {}
    for question in questions:
        q_docpartitions.update(generate_docpartition_per_question(docs_data_tensor, q_partitions, question))

    # embed sentences
    sents_embed_dir = embed_sentences(docs_data_tensor, sents_embed_dir, bertmodel, bert_config)

    # load model configs
    q_fold_config_map = {}
    for q in questions:
        mconfig, options = get_saved_config(config_path.format(q))
        argmax_indx = 'ignored'
        q_fold_config_map[q] = (mconfig, options, argmax_indx)

    results = run_predict(q_docpartitions, q_fold_config_map, bertmodel, state_dict_path, working_dir, sents_embed_dir,
                          to_gpu, gpu_index, num_epochs=1)
    return results
