import os
import itertools
from .utilities import get_device, create_directory, ReaderWriter, perfmetric_report, plot_loss
from .model import Attention, SentenceEncoder, DocEncoder, DocCategScorer, BertEmbedder, restrict_grad_
from .dataset import construct_load_dataloaders
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp


class HyperparamConfig:
    def __init__(self, encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size,
                 num_epochs):
        self.sentencoder_dim = encoder_dim
        self.num_layers = num_layers
        self.encoder_approach = encoder_approach
        self.attn_method = attn_method
        self.p_dropout = p_dropout
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def __repr__(self):
        desc = " sentencoder_dim:{}\n num_layers:{} \n encoder_approach:{} \n attn_method:{} \n p_dropout:{} \n " \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.sentencoder_dim, self.num_layers,
                                                                     self.encoder_approach, self.attn_method,
                                                                     self.p_dropout, self.l2_reg, self.batch_size,
                                                                     self.num_epochs)
        return desc


def generate_models_config(hyperparam_config, question, fold_num, fdtype):

    if(hyperparam_config.encoder_approach in {'[h_f;h_b]', '[h_f+h_b]'}):
        bidirection = True
    else:
        bidirection = False

    # determine docencoder input dim
    if(hyperparam_config.encoder_approach == '[h_f;h_b]'):
        docencoder_inputdim = 2*hyperparam_config.sentencoder_dim
        docencoder_hiddendim = docencoder_inputdim//4
    else:
        docencoder_inputdim = hyperparam_config.sentencoder_dim
        docencoder_hiddendim = docencoder_inputdim//2  # for now we divide the input vector dimension by 2

    bidirectional_concat_flag = False
    if(hyperparam_config.encoder_approach == '[h_f;h_b]'):
        attn_input_dim = 2*docencoder_hiddendim
        if(hyperparam_config.attn_method == 'additive'):
            bidirectional_concat_flag = True
    else:
        attn_input_dim = docencoder_hiddendim

    docscorer_input_dim = attn_input_dim

    # currently generic_config is shared across all models
    # leaving it as placeholder such that custom generic configs could be passed :)
    generic_config = {'fdtype': fdtype,
                      'bidirectional_concat_flag': bidirectional_concat_flag,
                      'encoder_approach': hyperparam_config.encoder_approach}

    sentencoder_config = {'input_dim': 768,  # bert model embedding dimension
                          'hidden_dim': hyperparam_config.sentencoder_dim,
                          'num_hiddenlayers': hyperparam_config.num_layers,
                          'bidirection': bidirection,
                          'pdropout': hyperparam_config.p_dropout,
                          'rnn_class': nn.GRU,
                          'nonlinear_fun': torch.relu,
                          'to_gpu': True,
                          'generic_config': generic_config}

    docencoder_config = {'input_dim': docencoder_inputdim,
                         'hidden_dim': docencoder_hiddendim,
                         'num_hiddenlayers': hyperparam_config.num_layers,
                         'bidirection': bidirection,
                         'pdropout': hyperparam_config.p_dropout,
                         'rnn_class': nn.GRU,
                         'nonlinear_fun': torch.relu,
                         'to_gpu': True,
                         'generic_config': generic_config}

    docscorer_config = {'input_dim': docscorer_input_dim}

    attnmodel_config = {'attn_method': hyperparam_config.attn_method,
                        'attn_input_dim': attn_input_dim,
                        'generic_config': generic_config}

    dataloader_config = {'batch_size': hyperparam_config.batch_size,
                         'num_workers': 0}

    bert_encoder_config = {'bert_train_flag': False,
                           'bert_all_output': False}

    config = {'dataloader_config': dataloader_config,
              'bert_encoder_config': bert_encoder_config,
              'sent_encoder_config': sentencoder_config,
              'doc_encoder_config': docencoder_config,
              'attnmodel_config': attnmodel_config,
              'doc_scorer_config': docscorer_config,
              'generic_config': generic_config}

    options = {'question': question,
               'fold_num': fold_num,
               'num_epochs': hyperparam_config.num_epochs,
               'weight_decay': hyperparam_config.l2_reg}

    return config, options


def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)


def run_neural_discern(data_partition, dsettypes, bertmodel, config, options, wrk_dir, sents_embed_dir,
                       state_dict_dir=None, to_gpu=True, gpu_index=0):
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, dsettypes, dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out = cld
    # print(class_weights)
    device = get_device(to_gpu, gpu_index)  # gpu device
    generic_config = config['generic_config']
    fdtype = generic_config['fdtype']
    if('train' in class_weights):
        class_weights = class_weights['train'].type(fdtype).to(device)  # update class weights to fdtype tensor
    else:
        class_weights = torch.tensor([1, 1]).type(fdtype).to(device)  # weighting all casess equally

    print("class weights", class_weights)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss
    rgrad_mode = options.get('restrict_grad_mode')  # we can add these to options
    rgrad_limit = options.get('restrict_grad_limit')

    num_epochs = options.get('num_epochs', 50)
    question = options.get('question') - 1
    fold_num = options.get('fold_num')

    # parse config dict
    bertencoder_config = config['bert_encoder_config']
    sentencoder_config = config['sent_encoder_config']
    docencoder_config = config['doc_encoder_config']
    attnmodel_config = config['attnmodel_config']
    docscorer_config = config['doc_scorer_config']

    # setup the models
    # bert model
    bert_encoder = BertEmbedder(bertmodel, bertencoder_config)
    bert_train_flag = bertencoder_config.get('bert_train_flag', False)
    bert_encoder.type(fdtype).to(device)

    # sentence encoder model
    sent_encoder = SentenceEncoder(sentencoder_config['input_dim'],
                                   sentencoder_config['hidden_dim'],
                                   num_hiddenlayers=sentencoder_config['num_hiddenlayers'],
                                   bidirection=sentencoder_config['bidirection'],
                                   pdropout=sentencoder_config['pdropout'],
                                   config=sentencoder_config['generic_config'],
                                   gpu_index=gpu_index)

    # doc encoder model
    attn_model = Attention(attnmodel_config['attn_method'],
                           attnmodel_config['attn_input_dim'],
                           config=attnmodel_config['generic_config'],
                           gpu_index=gpu_index)

    doc_encoder = DocEncoder(docencoder_config['input_dim'],
                             docencoder_config['hidden_dim'],
                             attn_model,
                             num_hiddenlayers=docencoder_config['num_hiddenlayers'],
                             bidirection=docencoder_config['bidirection'],
                             pdropout=docencoder_config['pdropout'],
                             config=docencoder_config['generic_config'],
                             gpu_index=gpu_index)

    # doc category scorer
    num_labels = len(class_weights)
    doc_categ_scorer = DocCategScorer(docscorer_config['input_dim'], num_labels)

    # define optimizer and group parameters
    models_param = list(sent_encoder.parameters()) + list(doc_encoder.parameters()) + \
        list(doc_categ_scorer.parameters())
    models = [(sent_encoder, 'sent_encoder'), (doc_encoder, 'doc_encoder'), (doc_categ_scorer, 'doc_categ_scorer')]
    if(bert_train_flag):
        models_param += list(bert_encoder.parameters())
        models += [(bert_encoder, 'bert_encoder')]

    if(state_dict_dir):  # load state dictionary of saved models
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

    # update models fdtype and move to device
    for m, m_name in models:
        m.type(fdtype).to(device)

    if('train' in data_loaders):
        weight_decay = options.get('weight_decay', 1e-3)
        optimizer = torch.optim.Adam(models_param, weight_decay=weight_decay, lr=1e-3)
        # see paper Cyclical Learning rates for Training Neural Networks for parameters' choice
        # `https://arxive.org/pdf/1506.01186.pdf`
        # pytorch version >1.1, scheduler should be called after optimizer
        # for cyclical lr scheduler, it should be called after each batch update
        num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
        c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
        base_lr = 3e-4
        max_lr = 5*base_lr  # 3-5 times base_lr
        cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                          mode='triangular', cycle_momentum=False)

    # store attention weights for validation and test set
    docid_attnweights_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'validation', 'test'}}
    # store sentences' attention weights

    if('validation' in data_loaders):
        m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))

    if(num_epochs > 1):
        fig_dir = create_directory(os.path.join(wrk_dir, 'figures'))

    # dump config dictionaries on disk
    config_dir = create_directory(os.path.join(wrk_dir, 'config'))
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))

    if(os.path.isfile(os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))):
        bert_proc_docs = ReaderWriter.read_data(os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))
        dump_embed_dict_flag = False
    else:
        bert_proc_docs = {}
        dump_embed_dict_flag = True

    # print(dump_embed_dict_flag)

    for epoch in range(num_epochs):
        # print("-"*35)
        for dsettype in dsettypes:
            print("device: {} | question: {} | fold_num: {} | epoch: {} | dsettype: {} | pid: {}"
                  "".format(device, options.get('question'), fold_num, epoch, dsettype, pid))
            pred_class = []
            ref_class = []

            data_loader = data_loaders[dsettype]
            # total_num_samples = len(data_loader.dataset)
            epoch_loss = 0.
            epoch_loss_deavrg = 0.

            if(dsettype == 'train'):  # should be only for train
                for m, m_name in models:
                    m.train()
            else:
                for m, m_name in models:
                    m.eval()

            sample_counter = 0
            for i_batch, samples_batch in enumerate(data_loader):
                # print('batch num:', i_batch)

                logprob_scores = []
                target_class = []
                # zero model grad
                if(dsettype == 'train'):
                    optimizer.zero_grad()

                docs_batch, docs_len, docs_sents_len, docs_attn_mask, docs_labels, docs_id = samples_batch

                docs_batch = docs_batch.to(device)
                docs_attn_mask = docs_attn_mask.to(device)
                docs_sents_len = docs_sents_len.type(torch.int64).numpy()  # to feed this in RNN
                docs_labels = docs_labels.type(torch.int64).to(device)

                with torch.set_grad_enabled(dsettype == 'train'):
                    # print("number of examples in batch:", docs_batch.size(0))
                    num_docs_perbatch = docs_batch.size(0)
                    for doc_indx in range(num_docs_perbatch):
                        # print('doc_indx', doc_indx)
                        doc_id = docs_id[doc_indx].item()
                        if(doc_id in bert_proc_docs):
                            # due to GPU limit
                            embed_sents = ReaderWriter.read_tensor(bert_proc_docs[doc_id], device=device)
                            # embed_sents = embed_sents.to(device)  # send to gpu device
                        else:
                            embed_sents = bert_encoder(docs_batch[doc_indx], docs_attn_mask[doc_indx],
                                                       docs_len[doc_indx].item())
                            # add embedding to dict
                            embed_fpath = os.path.join(sents_embed_dir, '{}.pkl'.format(doc_id))
                            ReaderWriter.dump_tensor(embed_sents, embed_fpath)
                            bert_proc_docs[doc_id] = embed_fpath

                        sents_rnn_hidden = sent_encoder(embed_sents, docs_sents_len[doc_indx],
                                                        docs_len[doc_indx].item())

                        # # remove the embedding from GPU
                        # bert_proc_docs[doc_id].to(cpu_device)
                        # print('sents_rnn_hidden', sents_rnn_hidden.shape)
                        enc_sents = sents_rnn_hidden
                        # print('enc_sents', enc_sents.shape)
                        doc_out, doc_attn_weights = doc_encoder(enc_sents)
                        # print('doc_out', doc_out.shape)
                        # print('doc_attn_weights', doc_attn_weights.shape)
                        # tracking attention weight for validation and test examples
                        if(dsettype in docid_attnweights_map):
                            docid_attnweights_map[dsettype][doc_id] = doc_attn_weights

                        logsoftmax_scores = doc_categ_scorer(doc_out)
                        __, pred_classindx = torch.max(logsoftmax_scores, 1)  # apply max on row level

                        # print('logsoftmax_scores', logsoftmax_scores.shape)
                        # print('pred_calssindx', pred_classindx.shape)
                        # print('ref labels', docs_labels[doc_indx, question].unsqueeze(0).shape)
                        # print('predicted class index:', pred_classindx.item())
                        # print('ref class index:', docs_labels[doc_indx, question].item())
                        pred_class.append(pred_classindx.item())
                        ref_class.append(docs_labels[doc_indx, question].item())

                        logprob_scores.append(logsoftmax_scores)
                        target_class.append(docs_labels[doc_indx, question].unsqueeze(0))
                        sample_counter += 1

                    # finished processing docs in batch
                    b_logprob_scores = torch.cat(logprob_scores, dim=0)
                    b_target_class = torch.cat(target_class, dim=0)
                    # print("b_logprob_scores", b_logprob_scores.shape)
                    # print("b_target_class", b_target_class.shape)
                    loss = loss_func(b_logprob_scores, b_target_class)
                    if(dsettype == 'train'):
                        # print("computing loss")
                        # backward step (i.e. compute gradients)
                        loss.backward()
                        # apply grad clipping
                        if(rgrad_mode):
                            restrict_grad_(models_param, rgrad_mode, rgrad_limit)
                        # optimzer step -- update weights
                        optimizer.step()
                        # after each batch step the scheduler
                        cyc_scheduler.step()
                    epoch_loss += loss.item()
                    # deaverage the loss to deal with last batch with unequal size
                    epoch_loss_deavrg += loss.item() * num_docs_perbatch

                    # do some cleaning -- get more GPU ;)
                    del docs_batch, docs_len, docs_sents_len, docs_attn_mask, docs_labels, docs_id
                    # torch.cuda.ipc_collect()
                    # torch.cuda.empty_cache()
            # end of epoch
            # print("+"*35)
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss_deavrg/len(data_loader.dataset))

            modelscore = perfmetric_report(pred_class, ref_class, epoch+1, flog_out[dsettype])
            perf = modelscore.macro_f1
            if(perf > score_dict[dsettype].macro_f1):
                score_dict[dsettype] = modelscore
                if(dsettype == 'validation'):
                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}.pkl'.format(m_name)))
                    # dump attention weights for the validation data for the best peforming model
                    dump_dict_content(docid_attnweights_map, ['validation'], 'docid_attnw_map', wrk_dir)
                elif(dsettype == 'test'):
                    # dump attention weights for the validation data
                    dump_dict_content(docid_attnweights_map, ['test'], 'docid_attnw_map', wrk_dir)

    if(num_epochs > 1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, fig_dir)

    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)
    if(dump_embed_dict_flag):
        print(bert_proc_docs)
        ReaderWriter.dump_data(bert_proc_docs, os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))
    return pred_class


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
            print(proc_articles_repr[docid]['sents_tok'][target_indx])
        print()


def return_attnw_over_sents(docid_attnweights_map, proc_articles_repr, topk=5):
    attended_sents = {}
    for docid in docid_attnweights_map:
        attended_sents[docid] = []
        attnw = docid_attnweights_map[docid]
        print('docid_attnweights_map: {}'.format(docid_attnweights_map))
        print('attnw: {}'.format(attnw))
        topk = topk if attnw.size(-1) > topk else attnw.size(-1)  # get top
        max_val, max_indx = torch.topk(attnw, topk, dim=1)
        for i in range(max_indx.size(-1)):
            target_indx = max_indx[0][i].item()
            sentence = proc_articles_repr[docid]['sents'][target_indx]
            attended_sents[docid].append({'sentence': sentence, 'weight': max_val[0][i].item()})
    return attended_sents


def validate_doc_attnw(docid_attnweights_map):
    for docid in docid_attnweights_map:
        attnw = docid_attnweights_map[docid]
        print(docid, 'sum:', attnw.sum(dim=1).item())
        print(docid, 'max:', torch.max(attnw, dim=1)[0].item())
        # the sum of attention weights should be close to 1)
        assert torch.allclose(attnw.sum(dim=1), torch.tensor([1.0], device=get_device(True)))
    print('passed validation test!!')


def generate_hyperparam_space():
    encoder_dim_vals = [128, 256, 512]  # can drop 512 if gpu crashes
    l2_reg_vals = [.1, .01, .001]
    encoder_approach_vals = ['[h_f;h_b]', '[h_f+h_b]']  # '[h_f]',
    num_layers_vals = [1, 2, 3]
    batch_size_vals = [4, 8, 16]
    dropout_vals = [0.1, 0.3, 0.4]
    attn_method_vals = ['additive', 'dot_scaled']
    num_epochs_vals = [25]
    hyperparam_space = list(itertools.product(*[encoder_dim_vals,  num_layers_vals, encoder_approach_vals,
                                                attn_method_vals, dropout_vals, l2_reg_vals, batch_size_vals,
                                                num_epochs_vals]))
    return hyperparam_space


def compute_numtrials(prob_interval_truemax, prob_estim):
    """ computes number of trials needed for random hyperparameter search
        see `algorithms for hyperparameter optimization paper
        <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__

        Args:
            prob_interval_truemax: float, probability interval of the true optimal hyperparam,
                i.e. within 5% expressed as .05
            prob_estim: float, probability/confidence level, i.e. 95% expressed as .95
    """
    n = np.log(1-prob_estim)/np.log(1-prob_interval_truemax)
    return(int(np.ceil(n))+1)


def get_hyperparam_options(prob_interval_truemax, prob_estim, random_seed=42):
    np.random.seed(random_seed)
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    hyperparam_space = generate_hyperparam_space()
    if(num_trials > len(hyperparam_space)):
        num_trials = len(hyperparam_space)
    indxs = np.random.choice(len(hyperparam_space), size=num_trials, replace=False)
    # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
    return [HyperparamConfig(*hyperparam_space[indx]) for indx in indxs]


def get_random_question_fold_per_hyperparam_exp(questions, random_seed=42):
    """Get for each question the fold number to use for identifying optimal hyperparams
    """
    np.random.seed(random_seed)
    q_fold = {}
    for q in questions:
        q_fold[q] = np.random.randint(5)
    return q_fold


def hyperparam_model_search(q_docpartitions, bertmodel, sents_embed_dir, root_dir, fdtype=torch.float32, num_epochs=15,
                            prob_interval_truemax=0.05, prob_estim=0.95, random_seed=42):
    questions = list(q_docpartitions.keys())
    # questions = [4]  # TODO: update this
    q_fold_map = get_random_question_fold_per_hyperparam_exp(questions, random_seed=random_seed)
    dsettypes = ['train', 'validation']
    for q, fold_num in q_fold_map.items():
        # get list of hyperparam configs
        hyperparam_options = get_hyperparam_options(prob_interval_truemax, prob_estim)
        data_partition = q_docpartitions[q][fold_num]
        # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
        for counter, hyperparam_config in enumerate(hyperparam_options):
            mconfig, options = generate_models_config(hyperparam_config, q, fold_num, fdtype)
            print("Running q{} hyperparam search #{}".format(q, counter))
            path = os.path.join(root_dir, 'question_{}'.format(q), 'fold_{}'.format(fold_num),
                                'config_{}'.format(counter))
            wrk_dir = create_directory(path)
            run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options, wrk_dir, sents_embed_dir)


def run_one_questions_hyperparam_search(queue, q, fold_num, data_partition, bertmodel, sents_embed_dir, root_dir,
                                        gpu_index, fdtype=torch.float32, num_epochs=15, prob_interval_truemax=0.05,
                                        prob_estim=0.95, random_seed=42):
    # get list of hyperparam configs
    hyperparam_options = get_hyperparam_options(prob_interval_truemax, prob_estim)
    dsettypes = ['train', 'validation']
    # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
    for counter, hyperparam_config in enumerate(hyperparam_options):
        mconfig, options = generate_models_config(hyperparam_config, q, fold_num, fdtype)
        print("Running q{} hyperparam search #{}".format(q, counter))
        wrk_dir = create_directory(os.path.join(root_dir, 'question_{}'.format(q), 'fold_{}'.format(fold_num),
                                                'config_{}'.format(counter)))
        run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options, wrk_dir, sents_embed_dir,
                           gpu_index=gpu_index)


def hyperparam_model_search_parallel(questions_to_run, q_docpartitions, bertmodel, sents_embed_dir, root_dir,
                                     question_gpu_map, fdtype=torch.float32, num_epochs=15, prob_interval_truemax=0.05,
                                     prob_estim=0.95, random_seed=42):
    q_fold_map = get_random_question_fold_per_hyperparam_exp(questions_to_run, random_seed=random_seed)
    queue = mp.Queue()
    q_processes = []

    # create a process for each question's hyperparam search
    for q, fold_num in q_fold_map.items():
        data_partition = q_docpartitions[q][fold_num]
        q_processes.append(mp.Process(target=run_one_questions_hyperparam_search, args=(queue, q, fold_num,
                                                                                        data_partition, bertmodel,
                                                                                        sents_embed_dir, root_dir,
                                                                                        question_gpu_map[q],
                                                                                        fdtype, num_epochs,
                                                                                        prob_interval_truemax,
                                                                                        prob_estim, random_seed)))

    for q_process in q_processes:
        print(">>> spawning hyperparam search process")
        q_process.start()

    for q_process in q_processes:
        q_process.join()
        print("<<< joined hyperparam search process")

    return


def get_saved_config(config_dir):
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options


def get_index_argmax(score_matrix, target_indx):
    argmax_indx = np.argmax(score_matrix, axis=0)[target_indx]
    return argmax_indx


def get_best_config_from_hyperparamsearch(questions, hyperparam_search_dir, num_trials=60, metric_indx=2):
    """Read best models config from all models tested in hyperparamsearch phase

    Args:
        questions: list, of questions [4,5,9,10,11]
        hyperparam_search_dir: string, path root directory where hyperparam models are stored
        num_trials: int, number of tested models (default 60 based on 0.05 interval and 0.95 confidence interval)
                    see :func: `compute_numtrials`
        metric_indx:int, (default 2) using macro_f1 as performance metric to evaluate among the tested models
    """
    # determine best config from hyperparam search
    q_fold_map = get_random_question_fold_per_hyperparam_exp(questions, random_seed=42)
    q_fold_config_map = {}
    for question, fold_num in q_fold_map.items():
        scores = np.ones((num_trials, 5))*-1
        exist_flag = False
        for config_num in range(num_trials):
            fold_dir = os.path.join(hyperparam_search_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))

            score_file = os.path.join(fold_dir, 'config_{}'.format(config_num), 'score_validation.pkl')
            if(os.path.isfile(score_file)):
                mscore = ReaderWriter.read_data(score_file)
                scores[config_num, 0] = mscore.best_epoch_indx
                scores[config_num, 1] = mscore.micro_f1
                scores[config_num, 2] = mscore.macro_f1
                scores[config_num, 3] = mscore.accuracy
                scores[config_num, 4] = mscore.auc
                exist_flag = True
        if(exist_flag):
            argmax_indx = get_index_argmax(scores, metric_indx)
            mconfig, options = get_saved_config(os.path.join(fold_dir, 'config_{}'.format(argmax_indx), 'config'))
            q_fold_config_map[question] = (mconfig, options, argmax_indx)
    return q_fold_config_map


def train_val_run(q_docpartitions, q_fold_config_map, bertmodel, train_val_dir, sents_embed_dir, question_gpu_map,
                  num_epochs=25, max_folds=None):
    dsettypes = ['train', 'validation']
    for question in q_fold_config_map:
        mconfig, options, __ = q_fold_config_map[question]
        options['num_epochs'] = num_epochs  # override number of epochs using user specified value
        for fold_num in q_docpartitions[question]:
            if max_folds is None or fold_num < max_folds:
                # update options fold num to the current fold
                options['fold_num'] = fold_num
                data_partition = q_docpartitions[question][fold_num]
                path = os.path.join(train_val_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))
                wrk_dir = create_directory(path)
                run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options, wrk_dir, sents_embed_dir,
                                   gpu_index=question_gpu_map[question])


def train_val_run_one_question(queue, question, q_docpartitions, q_fold_config_map, bertmodel, train_val_dir,
                               sents_embed_dir, gpu_index, num_epochs=25, max_folds=None):
    dsettypes = ['train', 'validation']
    mconfig, options, __ = q_fold_config_map[question]
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    for fold_num in q_docpartitions[question]:
        if max_folds is None or fold_num < max_folds:
            # update options fold num to the current fold
            options['fold_num'] = fold_num
            data_partition = q_docpartitions[question][fold_num]
            path = os.path.join(train_val_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))
            wrk_dir = create_directory(path)
            run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options, wrk_dir, sents_embed_dir,
                               gpu_index=gpu_index)


def test_run(q_docpartitions, q_fold_config_map, bertmodel, train_val_dir, test_dir, sents_embed_dir, gpu_index,
             num_epochs=1):
    dsettypes = ['test']
    for question in q_fold_config_map:
        mconfig, options, __ = q_fold_config_map[question]
        options['num_epochs'] = num_epochs  # override number of epochs using user specified value
        for fold_num in q_docpartitions[question]:
            # update options fold num to the current fold
            options['fold_num'] = fold_num
            data_partition = q_docpartitions[question][fold_num]
            path = os.path.join(train_val_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))
            # only run testing on question and folds that were run (may not all have been run in code test mode)
            if os.path.exists(path):
                train_dir = create_directory(path)
                # load state_dict pth
                state_dict_pth = os.path.join(train_dir, 'model_statedict')

                path = os.path.join(test_dir, 'question_{}'.format(question), 'fold_{}'.format(fold_num))
                test_wrk_dir = create_directory(path)

                run_neural_discern(data_partition, dsettypes, bertmodel, mconfig, options, test_wrk_dir,
                                   sents_embed_dir, state_dict_dir=state_dict_pth, gpu_index=gpu_index)

# ==================================================


def predict_neural_discern(data_partition, bertmodel, config, options, wrk_dir, sents_embed_dir,
                           state_dict_dir=None, to_gpu=True, gpu_index=0):
    dsettype = 'test'
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, [dsettype], dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out = cld
    # print(class_weights)
    device = get_device(to_gpu, gpu_index)  # gpu device
    generic_config = config['generic_config']
    fdtype = generic_config['fdtype']

    class_weights = torch.tensor([1, 1]).type(fdtype).to(device)  # weighting all casess equally

    print("class weights", class_weights)

    fold_num = options.get('fold_num')

    # parse config dict
    bertencoder_config = config['bert_encoder_config']
    sentencoder_config = config['sent_encoder_config']
    docencoder_config = config['doc_encoder_config']
    attnmodel_config = config['attnmodel_config']
    docscorer_config = config['doc_scorer_config']

    # setup the models
    # bert model
    bert_encoder = BertEmbedder(bertmodel, bertencoder_config)
    bert_encoder.type(fdtype).to(device)

    # sentence encoder model
    sent_encoder = SentenceEncoder(sentencoder_config['input_dim'],
                                   sentencoder_config['hidden_dim'],
                                   num_hiddenlayers=sentencoder_config['num_hiddenlayers'],
                                   bidirection=sentencoder_config['bidirection'],
                                   pdropout=sentencoder_config['pdropout'],
                                   config=sentencoder_config['generic_config'],
                                   gpu_index=gpu_index)

    # doc encoder model
    attn_model = Attention(attnmodel_config['attn_method'],
                           attnmodel_config['attn_input_dim'],
                           config=attnmodel_config['generic_config'],
                           gpu_index=gpu_index)

    doc_encoder = DocEncoder(docencoder_config['input_dim'],
                             docencoder_config['hidden_dim'],
                             attn_model,
                             num_hiddenlayers=docencoder_config['num_hiddenlayers'],
                             bidirection=docencoder_config['bidirection'],
                             pdropout=docencoder_config['pdropout'],
                             config=docencoder_config['generic_config'],
                             gpu_index=gpu_index)

    # doc category scorer
    num_labels = len(class_weights)
    doc_categ_scorer = DocCategScorer(docscorer_config['input_dim'], num_labels)

    # define optimizer and group parameters
    models = [(sent_encoder, 'sent_encoder'), (doc_encoder, 'doc_encoder'), (doc_categ_scorer, 'doc_categ_scorer')]

    if(state_dict_dir):  # load state dictionary of saved models
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

    # update models fdtype and move to device
    for m, m_name in models:
        m.type(fdtype).to(device)

    # store attention weights for validation and test set
    docid_attnweights_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'validation', 'test'}}
    # store sentences' attention weights

    if(os.path.isfile(os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))):
        bert_proc_docs = ReaderWriter.read_data(os.path.join(sents_embed_dir, 'bert_proc_docs.pkl'))
    else:
        bert_proc_docs = {}

    print("device: {} | question: {} | fold_num: {} | dsettype: {} | pid: {}"
          "".format(device, options.get('question'), fold_num, dsettype, pid))
    pred_class = []

    data_loader = data_loaders[dsettype]

    for m, m_name in models:
        m.eval()

    sample_counter = 0
    for i_batch, samples_batch in enumerate(data_loader):

        logprob_scores = []

        docs_batch, docs_len, docs_sents_len, docs_attn_mask, docs_labels, docs_id = samples_batch

        docs_batch = docs_batch.to(device)
        docs_attn_mask = docs_attn_mask.to(device)
        docs_sents_len = docs_sents_len.type(torch.int64).numpy()  # to feed this in RNN
        docs_labels = docs_labels.type(torch.int64).to(device)

        with torch.set_grad_enabled(dsettype == 'train'):
            # print("number of examples in batch:", docs_batch.size(0))
            num_docs_perbatch = docs_batch.size(0)
            for doc_indx in range(num_docs_perbatch):
                # print('doc_indx', doc_indx)
                doc_id = docs_id[doc_indx].item()
                if(doc_id in bert_proc_docs):
                    # due to GPU limit
                    embed_sents = ReaderWriter.read_tensor(bert_proc_docs[doc_id], device)
                else:
                    embed_sents = bert_encoder(docs_batch[doc_indx], docs_attn_mask[doc_indx],
                                               docs_len[doc_indx].item())
                    # add embedding to dict
                    embed_fpath = os.path.join(sents_embed_dir, '{}.pkl'.format(doc_id))
                    ReaderWriter.dump_tensor(embed_sents, embed_fpath)
                    bert_proc_docs[doc_id] = embed_fpath

                sents_rnn_hidden = sent_encoder(embed_sents, docs_sents_len[doc_indx],
                                                docs_len[doc_indx].item())

                # # remove the embedding from GPU
                # bert_proc_docs[doc_id].to(cpu_device)
                # print('sents_rnn_hidden', sents_rnn_hidden.shape)
                enc_sents = sents_rnn_hidden
                # print('enc_sents', enc_sents.shape)
                doc_out, doc_attn_weights = doc_encoder(enc_sents)
                # print('doc_out', doc_out.shape)
                # print('doc_attn_weights', doc_attn_weights.shape)
                # tracking attention weight for validation and test examples
                if(dsettype in docid_attnweights_map):
                    docid_attnweights_map[dsettype][doc_id] = doc_attn_weights

                logsoftmax_scores = doc_categ_scorer(doc_out)
                __, pred_classindx = torch.max(logsoftmax_scores, 1)  # apply max on row level

                pred_class.append(pred_classindx.item())

                logprob_scores.append(logsoftmax_scores)
                sample_counter += 1

            # log probabilities
            b_logprob_scores = torch.cat(logprob_scores, dim=0)

            # do some cleaning -- get more GPU ;)
            del docs_batch, docs_len, docs_sents_len, docs_attn_mask, docs_labels, docs_id

    # end of epoch
    return {'pred_class': pred_class,
            'logprob_score_class0': [t.data.cpu().numpy()[0] for t in b_logprob_scores],
            'logprob_score_class1': [t.data.cpu().numpy()[1] for t in b_logprob_scores],
            'attention_weight_map': docid_attnweights_map['test']}
