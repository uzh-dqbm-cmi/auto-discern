#!/usr/bin/env python
# coding: utf-8

# % load_ext autoreload
# % autoreload 2

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import datetime
import pandas as pd
import torch
from pytorch_pretrained_bert import BertModel

from neural.data_processor import DataDictProcessor
from neural.dataset import generate_docpartition_per_question, validate_q_docpartitions, compute_class_weights_per_fold_
from neural.model import BertEmbedder, generate_sents_embeds_from_docs
from neural.run_workflow import generate_models_config, HyperparamConfig, hyperparam_model_search_parallel, \
    get_best_config_from_hyperparamsearch, train_val_run, train_val_run_one_question, test_run
from neural.utilities import ReaderWriter, create_directory


def read_pickles(directory):
    # processed data directory
    data_dir = create_directory('proc_data_uncased', directory)

    # Read stored data structures
    # {question:{fold_num:{dsettype:np.array(list of doc_ids)}}}
    q_partitions = ReaderWriter.read_data(os.path.join(data_dir, 'q_partitions.pkl'))
    # instance of :class:`DocDataTensor`
    docs_data_tensor = ReaderWriter.read_data(os.path.join(data_dir, 'docs_data_tensor.pkl'))
    # instance variable for processed articles dict
    proc_articles_dict = ReaderWriter.read_data(os.path.join(data_dir, 'processor_articles_dict.pkl'))
    # instance variable for computed articles reprs
    proc_articles_repr = ReaderWriter.read_data(os.path.join(data_dir, 'processor_articles_repr.pkl'))
    proc_config = ReaderWriter.read_data(os.path.join(data_dir, 'processor_config.pkl'))

    return q_partitions, docs_data_tensor, proc_articles_dict, proc_articles_repr, proc_config


def build_doc_partitions(questions, proc_articles_repr, proc_articles_dict, proc_config, docs_data_tensor, q_partitions,
                         validate=False):
    processor = DataDictProcessor(proc_config)
    # reassign updated articles_dict and articles_repr to processor instance
    processor.set_instance_attr(proc_articles_repr, proc_articles_dict, proc_config)

    # turn the list of ids in q_partitions into dataset using PartitionDataTensor
    q_docpartitions = {}
    for question in questions:
        q_docpartitions.update(generate_docpartition_per_question(docs_data_tensor, q_partitions, question))

    if validate:
        validate_q_docpartitions(q_docpartitions, q_partitions)

    # add class weights in q_docpartitions
    # we will use the weights to emphasize less occuring class during training phase
    compute_class_weights_per_fold_(q_docpartitions)

    return q_docpartitions


def write_sents_embeddings(directory, bertmodel, sents_embed_dir_name, docs_data_tensor):
    # === Generate sents embedding ===
    # load BertModel

    # define BertEmbedder
    bert_config = {'bert_train_flag': False,
                   'bert_all_output': False}
    bertembeder = BertEmbedder(bertmodel, bert_config)
    sents_embed_dir = create_directory(sents_embed_dir_name, directory)
    fdtype = torch.float32

    # generate and dump bert embedding for the tokens inside the specificed embedding directory
    bert_proc_docs = generate_sents_embeds_from_docs(docs_data_tensor, bertembeder, sents_embed_dir, fdtype)
    ReaderWriter.dump_data(bert_proc_docs, os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))


def read_sents_embeddings(directory, sents_embed_dir_name):
    ''' read the 3-dimensional doc tensors
    doc x sent x embedding'''

    sents_embed_dir = create_directory(sents_embed_dir_name, directory)
    bert_proc_docs = ReaderWriter.read_data(os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))
    return bert_proc_docs, sents_embed_dir


def run_hyperparam_search(questions_to_run, directory, q_docpartitions, bertmodel, sents_embed_dir, question_gpu_map):
    # TODO: parallelize first!
    hyperparam_search_dir = create_directory('hyperparam_search', directory)
    hyperparam_model_search_parallel(questions_to_run, q_docpartitions, bertmodel, sents_embed_dir,
                                     hyperparam_search_dir,
                                     question_gpu_map,
                                     fdtype=torch.float32,
                                     num_epochs=15,
                                     prob_interval_truemax=0.05,
                                     prob_estim=0.95, random_seed=42)


def run_training_parallel(questions_to_run, directory, q_docpartitions, q_config_map, bertmodel, sents_embed_dir,
                          question_gpu_map, num_epochs, max_folds):
    train_val_dir = create_directory('train_validation', directory)
    queue = mp.Queue()
    q_processes = []
    # create a process for each question model
    for q in questions_to_run:
        q_processes.append(mp.Process(target=train_val_run_one_question, args=(queue, q, q_docpartitions, q_config_map,
                                                                               bertmodel, train_val_dir,
                                                                               sents_embed_dir, question_gpu_map[q],
                                                                               num_epochs, max_folds)))

    for q_process in q_processes:
        print(">>> spawning process")
        q_process.start()

    for q_process in q_processes:
        q_process.join()
        print("<<< joined process")

    return train_val_dir


def run_training(directory, q_docpartitions, q_config_map, bertmodel, sents_embed_dir, question_gpu_map, num_epochs,
                 max_folds):
    train_val_dir = create_directory('train_validation', directory)
    train_val_run(q_docpartitions, q_config_map, bertmodel, train_val_dir, sents_embed_dir, question_gpu_map,
                  num_epochs, max_folds)
    return train_val_dir


def evaluate_on_test_set(directory, q_docpartitions, q_config_map, bertmodel, train_val_dir, sents_embed_dir,
                         gpu_index):
    test_dir = create_directory('test', directory)
    test_run(q_docpartitions, q_config_map, bertmodel, train_val_dir, test_dir, sents_embed_dir, gpu_index,
             num_epochs=1)
    return test_dir


def get_performance_results(question, target_dir, num_folds, dsettype):
    all_perf = {}
    num_metrics = 3
    perf_dict = [{} for i in range(num_metrics)]  # track micro_f1, macro_f1, accuracy
    question_name = ['q{}'.format(question)]
    for fold_num in range(num_folds):

        fold_dir = os.path.join(target_dir,
                                'question_{}'.format(question),
                                'fold_{}'.format(fold_num))

        score_file = os.path.join(fold_dir, 'score_{}.pkl'.format(dsettype))
        if os.path.isfile(score_file):
            mscore = ReaderWriter.read_data(score_file)
            perf_dict[0]['fold{}'.format(fold_num)] = mscore.micro_f1
            perf_dict[1]['fold{}'.format(fold_num)] = mscore.macro_f1
            perf_dict[2]['fold{}'.format(fold_num)] = mscore.accuracy
    perf_df = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf, index=question_name)
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df.append(all_perf_df.sort_values('mean', ascending=False))
    return perf_df


def build_accuracy_dfs(q_docpartitions, test_dir):
    micro_f1_df = pd.DataFrame()
    macro_f1_df = pd.DataFrame()
    accuracy_df = pd.DataFrame()

    for q in q_docpartitions:
        micro_f1, macro_f1, accuracy = get_performance_results(q, test_dir, 5, 'test')
        micro_f1_df = pd.concat([micro_f1_df, micro_f1], sort=True)
        macro_f1_df = pd.concat([macro_f1_df, macro_f1], sort=True)
        accuracy_df = pd.concat([accuracy_df, accuracy], sort=True)

    return micro_f1_df, macro_f1_df, accuracy_df


def highlight_attnw_over_sents(docid_attnweights_map, proc_articles_repr, topk=5):
    for docid in docid_attnweights_map:
        attnw = docid_attnweights_map[docid]
        topk = topk if attnw.size(-1) > topk else attnw.size(-1)  # get top
        max_val, max_indx = torch.topk(attnw, topk, dim=1)
        print(docid)
        print("attended sent:")
        for i in range(max_indx.size(-1)):
            target_indx = max_indx[0][i].item()
            print("sentence num:", target_indx, "attnw:", max_val[0][i].item())
            print(proc_articles_repr[docid]['sents'][target_indx])
        print()


def verbose_print(text, print_yn):
    if print_yn:
        print(text)

# ============================================================


if __name__ == '__main__':

    rewrite_sentence_embeddings = False
    run_hyper_param_search = True
    questions_to_run = [4, 5, 9, 10, 11]
    max_folds = 5
    num_epochs = 25
    verbose = True
    experiment_to_rerun = None

    questions = (4, 5, 9, 10, 11)
    question_gpu_map = {
        4: 1,
        5: 2,
        9: 3,
        10: 4,
        11: 5
    }
    print("Running questions: {} | folds: {} | epochs: {}".format(questions_to_run, max_folds, num_epochs))

    base_dir = '/opt/data/autodiscern/aa_neural'
    if experiment_to_rerun is None:
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_dir = os.path.join(base_dir, 'experiments', time_stamp)
        create_directory(exp_dir)
    else:
        exp_dir = os.path.join(base_dir, 'experiments', experiment_to_rerun)

    bert_model_dir = '/opt/data/autodiscern/aa_neural/aws_downloads/bert-base-uncased.tar.gz'
    sents_embed_dir_name = 'sents_bert_embed_uncased'

    # ---

    verbose_print("Loading objects...", verbose)

    q_partitions, docs_data_tensor, proc_articles_dict, proc_articles_repr, proc_config = read_pickles(base_dir)
    q_docpartitions = build_doc_partitions(questions, proc_articles_repr, proc_articles_dict, proc_config,
                                           docs_data_tensor, q_partitions)

    bertmodel = BertModel.from_pretrained(bert_model_dir)

    if rewrite_sentence_embeddings:
        verbose_print("Writing sentence embeddings...", verbose)
        write_sents_embeddings(base_dir, bertmodel, sents_embed_dir_name, docs_data_tensor)

    verbose_print("Reading sentence embeddings...", verbose)
    bert_proc_docs, sents_embed_dir = read_sents_embeddings(base_dir, sents_embed_dir_name)

    if run_hyper_param_search:
        verbose_print("Running hyper-parameter search...", verbose)
        run_hyperparam_search(questions_to_run, exp_dir, q_docpartitions, bertmodel, sents_embed_dir, question_gpu_map)
        hyperparam_search_dir = create_directory('hyperparam_search', exp_dir)
        q_config_map = get_best_config_from_hyperparamsearch(questions, hyperparam_search_dir, num_trials=60,
                                                             metric_indx=2)
    else:
        verbose_print("Creating custom hyper-parameter config...", verbose)
        # using custom hyperparam configuration for all questions
        # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
        hyperparam_config = HyperparamConfig(256, 2, '[h_f+h_b]', 'additive', 0.3, 0.01, 16, 25)
        q_config_map = {}
        fold_num = -1
        for q in questions:
            if q in questions_to_run:
                mconfig, options = generate_models_config(hyperparam_config, q, fold_num, torch.float32)
                q_config_map[q] = (mconfig, options, -1)

    verbose_print("Training...", verbose)

    train_val_dir = run_training_parallel(questions_to_run, exp_dir, q_docpartitions, q_config_map, bertmodel,
                                          sents_embed_dir, question_gpu_map, num_epochs, max_folds)

    # train_val_dir = run_training(exp_dir, q_docpartitions, q_config_map, bertmodel, sents_embed_dir,
    #                              question_gpu_map, num_epochs, max_folds)

    verbose_print("Evaluating on test set...", verbose)
    test_dir = evaluate_on_test_set(exp_dir, q_docpartitions, q_config_map, bertmodel, train_val_dir, sents_embed_dir,
                                    gpu_index=1)

    micro_f1_df, macro_f1_df, accuracy_df = build_accuracy_dfs(q_docpartitions, test_dir)
    print("micro_f1_df: {}".format(micro_f1_df))
    print("macro_f1_df: {}".format(macro_f1_df))
    print("accuracy_df: {}".format(accuracy_df))

    # extra results
    # docid_attnw_map_val = ReaderWriter.read_data(os.path.join(test_dir, 'question_10', 'fold_2',
    #                                                           'docid_attnw_map_test.pkl')
    #                                              )
    # highlight_attnw_over_sents(docid_attnw_map_val, proc_articles_repr, topk=5)
    # validate_doc_attnw(docid_attnw_map_val)
