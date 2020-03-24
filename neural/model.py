import os
from .utilities import get_device, ReaderWriter
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Attention(nn.Module):
    def __init__(self, attn_method, input_dim, nonlinear_func=torch.tanh, config={}, to_gpu=True, gpu_index=0):
        '''
        Args:
            attn_method: string, {'additive', 'dot', 'dot_scaled'}
            input_dim: int, size of the input vector (i.e. sentence vector representation)

        '''

        super(Attention, self).__init__()
        self.attn_method = attn_method
        self.input_dim = input_dim
        self.nonlinear_func = nonlinear_func
        self.device = get_device(to_gpu, gpu_index)

        # print("Modified attention model")
        # print("method: ", self.attn_method)
        # print("input dim:", self.input_dim)
        # print("generic config:", config)

        self.fdtype = config.get('fdtype', torch.float32)
        self.bidirectional_concat_flag = config.get('bidirectional_concat_flag', False)
        if(self.bidirectional_concat_flag):
            input_divider = 2
        else:
            input_divider = 1

        # print("input divider:", input_divider)
        if(self.attn_method == 'additive'):
            self.attnW = nn.Linear(self.input_dim, self.input_dim//input_divider)
            queryv_dim = self.input_dim//input_divider  # we use the mapped vector size
        elif(self.attn_method in {'dot', 'dot_scaled'}):  # only dot prodcut operation
            queryv_dim = self.input_dim  # we use the input vector size since we will perform dot product
            if(self.attn_method == 'dot_scaled'):
                self.scaler = torch.sqrt(torch.tensor(queryv_dim, dtype=self.fdtype, device=self.device))
        # use this as query vector against the encoder outputs
        self.queryv = nn.Parameter(torch.randn(queryv_dim, dtype=self.fdtype, device=self.device), requires_grad=True)

    def forward(self, encoder_outputs):
        '''Performs forward computation

        Args:
            encoder_outputs: torch.Tensor, (1, sents, encoding_dim), dtype=torch.float32
        '''

        if(self.attn_method == 'additive'):
            # do the mapping using one-layer mlp network, followed by nonlinear element-wise operation
            encoder_map = self.nonlinear_func(self.attnW(encoder_outputs))
            # print('encoder_map size', encoder_map.size())
            # print('queryv size', self.queryv.size())
        else:
            encoder_map = encoder_outputs
            # print('encoder_map size', encoder_map.size())
            # print('queryv size', self.queryv.size())
        # using  matmul to compute tensor vector multiplication
        attn_weights = encoder_map.matmul(self.queryv)
        if(self.attn_method == 'dot_scaled'):
            attn_weights = attn_weights/self.scaler
        # softmax
        attn_weights_norm = torch.softmax(attn_weights, dim=1)

        # returns (1, sents)
        return attn_weights_norm


class BertEmbedder(nn.Module):

    def __init__(self, bertmodel, proc_config):
        super(BertEmbedder, self).__init__()
        self.bertmodel = bertmodel.float()
        self.config = proc_config

        # use BertModel as word embedder
        self.bert_train_flag = self.config.get('bert_train_flag', False)
        # for now we are taking the last layer hidden vectors
        self.bert_all_output = self.config.get('bert_all_output', False)

        if not self.bert_train_flag:
            self.bertmodel.eval()
        else:
            self.bertmodel.train()

    # def forward_batch_sents(self, doc_tensor, attention_mask, num_sents):
    #     '''

    #     Args:
    #         doc_tenosr: tensor, (sents, max_sent_len)
    #         attention_mask: tensor, (sents, max_sent_len)
    #         num_sents: int, actual number of sentences in the document

    #     TODO: add flags and logic to handle multiple layers embedding (i.e. 12 layers embedding)
    #     '''
    #     # use BertModel as word embedder
    #     bert_train_flag = self.config.get('bert_train_flag', False)
    #     # for now we are taking the last layer hidden vectors
    #     bert_all_output = self.config.get('bert_all_output', False)
    #     bertmodel = self.bertmodel
    #     if(not bert_train_flag):
    #         bertmodel.eval()
    #     else:
    #         bertmodel.train()

    #     with torch.set_grad_enabled(bert_train_flag):
    #         encoded_layers, __ = bertmodel(doc_tensor[:num_sents],
    #                                        attention_mask=attention_mask[:num_sents],
    #                                        output_all_encoded_layers=bert_all_output)
    #         print('encoded_layers shape', encoded_layers.shape)

    #     # print("finished embedding sents using BERT!")
    #     return encoded_layers

    def forward(self, doc_tensor, attention_mask, num_sents):
        '''

        Args:
            doc_tenosr: tensor, (sents, max_sent_len)
            attention_mask: tensor, (sents, max_sent_len)
            num_sents: int, actual number of sentences in the document

        TODO: add flags and logic to handle multiple layers embedding (i.e. 12 layers embedding)
        '''

        embed_layers_lst = []

        for sent_indx in range(num_sents):  # going over each sentenece one by one due to GPU limit :((
            # print('sent_indx', sent_indx)
            with torch.set_grad_enabled(self.bert_train_flag):
                encoded_layers, __ = self.bertmodel(doc_tensor[sent_indx:sent_indx+1],
                                                    attention_mask=attention_mask[sent_indx:sent_indx+1],
                                                    output_all_encoded_layers=self.bert_all_output)
                embed_layers_lst.append(encoded_layers)

        # # embed all sentences at once
        # with torch.set_grad_enabled(self.bert_train_flag):
        #     encoded_layers, __ = self.bertmodel(doc_tensor, attention_mask=attention_mask,
        #                                         output_all_encoded_layers=self.bert_all_output)
        #
        # return encoded_layers

        # concat everything
        out = torch.cat(embed_layers_lst, dim=0)
        # print("finished embedding sents using BERT!")
        return out


class SentenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hiddenlayers=1,
                 bidirection=False, pdropout=0.1, rnn_class=nn.GRU,
                 nonlinear_func=torch.relu, config={}, to_gpu=True, gpu_index=0):

        super(SentenceEncoder, self).__init__()
        self.input_dim = input_dim  # embedding dimension as input to RNN
        self.device = get_device(to_gpu, gpu_index)
        self.hidden_dim = hidden_dim  # dimension of RNN output
        self.num_hiddenlayers = num_hiddenlayers
        self.pdropout = pdropout
        self.dropout_layer = nn.Dropout(pdropout)
        self.config = config
        self.fdtype = self.config.get('fdtype', torch.float32)

        if(bidirection):
            self.num_directions = 2
        else:
            self.num_directions = 1

        # RNN module inserts dropout between layers of RNN except for the output of the last layer!
        if(self.num_hiddenlayers == 1 and self.pdropout > 0):
            rnn_dropout = 0
        else:
            rnn_dropout = self.pdropout
        self.rnn = rnn_class(self.input_dim, self.hidden_dim, num_layers=self.num_hiddenlayers,
                             dropout=rnn_dropout, bidirectional=bidirection, batch_first=True)
        self.nonlinear_func = nonlinear_func

    def init_hidden(self, batch_size):
        """initialize hidden vectors at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """
        # a hidden vector has the shape (num_layers*num_directions, batch, hidden_dim)

        h0 = torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).to(device=self.device,
                                                                                                    dtype=self.fdtype)
        if(isinstance(self.rnn, nn.LSTM)):
            c0 = torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim
                             ).to(device=self.device, dtype=self.fdtype)
            hiddenvec = (h0, c0)
        else:
            hiddenvec = h0
        return(hiddenvec)

    def _process_rnn_hidden_output(self, hidden):
        encoder_approach = self.config.get('encoder_approach')
        lastlayer_indx = -1
        batch_size = hidden.size(1)

        hn = hidden.view(self.num_hiddenlayers, self.num_directions, batch_size, self.hidden_dim)
        if(encoder_approach == '[h_f]'):  # keep only the last forward hidden state vector
            return hn[lastlayer_indx]  # (1, num_sents, hidden_dim)
        elif(encoder_approach == '[h_f+h_b]'):
            return hn[lastlayer_indx].sum(dim=0).unsqueeze(0)  # (1, num_sents, hidden_dim)
        elif(encoder_approach == '[h_f;h_b]'):
            frwd_indx = 0
            bckwd_indx = 1
            # (1, num_sents, 2*hidden_dim)
            hn = torch.cat([hn[lastlayer_indx, frwd_indx, :, :], hn[lastlayer_indx, bckwd_indx, :, :]], dim=-1)
            return hn.unsqueeze(0)

    def _run_rnn(self, embed_sents, doc_sents_len, num_sents):
        # apply dropout
        embed_sents = self.dropout_layer(embed_sents)
        # init hidden
        hidden = self.init_hidden(num_sents)
        # pack the batch
        packed_embeds = pack_padded_sequence(embed_sents, doc_sents_len[:num_sents], batch_first=True,
                                             enforce_sorted=False)
        # print("packed_embeds", "\n", packed_embeds)
        packed_rnn_out, hidden = self.rnn(packed_embeds, hidden)
        # print("packed_rnn_out", "\n", packed_rnn_out)
        # print("hidden", "\n", hidden)
        # we need to unpack sequences
        # unpacked_output, out_seqlen = pad_packed_sequence(packed_rnn_out, batch_first=True)
        # return unpacked_output, hidden
        return self._process_rnn_hidden_output(hidden)

    # def _process_rnn_hidden_output(self, unpacked_out, hidden):
    #     encoder_approach = self.config.get('encoder_approach')
    #     lastlayer_indx = -1
    #     batch_size = hidden.size(1)
    #     hn = hidden.view(self.num_hiddenlayers, self.num_directions, batch_size, self.hidden_dim)
    #     if(encoder_approach == '[h_f]'):  # keep only the last forward hidden state vector
    #         return hn[lastlayer_indx]  # (1, num_sents, hidden_dim)
    #     else:
    #         # num_sents, max_num_tokens, num_directions, hidden_dim
    #         rnn_out = unpacked_out.view(unpacked_out.size(0), unpacked_out.size(1), self.num_directions,
    #                                     self.hidden_dim)
    #         frwd_indx = 0
    #         bckwd_indx = 1
    #         t0 = 0
    #         if(encoder_approach == '[h_f+h_b]'):
    #             res = hn[lastlayer_indx, frwd_indx, :, :] + rnn_out[:, t0, bckwd_indx, :]
    #             return res.unsqueeze(0)  # (1, num_sents, hidden_dim)
    #         elif(encoder_approach == '[h_f;h_b]'):
    #             # (1, num_sents, 2*hidden_dim)
    #             hn = torch.cat([hn[lastlayer_indx, frwd_indx, :, :],  rnn_out[:, t0, bckwd_indx, :]], dim=-1)
    #             return hn.unsqueeze(0)

    # def _run_rnn(self, embed_sents, doc_sents_len, num_sents):
    #     # apply dropout
    #     embed_sents = self.dropout_layer(embed_sents)
    #     # init hidden
    #     hidden = self.init_hidden(num_sents)
    #     # pack the batch
    #     packed_embeds = pack_padded_sequence(embed_sents, doc_sents_len[:num_sents], batch_first=True,
    #                                          enforce_sorted=False)
    #     # print("packed_embeds", "\n", packed_embeds)
    #     packed_rnn_out, hidden = self.rnn(packed_embeds, hidden)
    #     # print("packed_rnn_out", "\n", packed_rnn_out)
    #     # print("hidden", "\n", hidden)
    #     # we need to unpack sequences
    #     unpacked_output, out_seqlen = pad_packed_sequence(packed_rnn_out, batch_first=True)
    #     # return unpacked_output, hidden
    #     return self._process_rnn_hidden_output(unpacked_output, hidden)

    def forward(self, embed_sents, doc_sents_len, num_sents):
        """ perform forward computation

            Args:
                embed_sents: torch.Tensor, (sents, max_sent_len, embed_dim), dtype=torch.float32 or torch.float64
                    depending on fdtype.
                doc_sents_len: torch.Tensor, (sents, ), dtype=torch.int64
                num_sents: int, actual number of sentences in the doc
        """

        return self._run_rnn(embed_sents, doc_sents_len, num_sents)


class DocEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_model,
                 num_hiddenlayers=1, bidirection=False, pdropout=0.1,
                 rnn_class=nn.GRU, nonlinear_func=torch.relu, config={}, to_gpu=True, gpu_index=0):

        super(DocEncoder, self).__init__()
        self.device = get_device(to_gpu, gpu_index)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # dimension of the hidden vector from rnn
        self.attn_model = attn_model  # instance of :class:`Attention`
        self.num_hiddenlayers = num_hiddenlayers
        self.pdropout = pdropout
        self.dropout_layer = nn.Dropout(pdropout)

        self.config = config
        # to get options for the attention module
        self.fdtype = self.config.get('fdtype', torch.float32)

        if(bidirection):
            self.num_directions = 2
        else:
            self.num_directions = 1

        # rnn module inserts dropout between layers of rnn except for the output of the last layer!
        if(self.num_hiddenlayers == 1 and self.pdropout > 0):
            rnn_dropout = 0
        else:
            rnn_dropout = self.pdropout

        self.rnn = rnn_class(self.input_dim, self.hidden_dim, num_layers=self.num_hiddenlayers, dropout=rnn_dropout,
                             bidirectional=bidirection, batch_first=True)

        self.nonlinear_func = nonlinear_func

    def init_hidden(self, batch_size):
        """initialize hidden vectors at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """
        # a hidden vector has the shape (num_layers*num_directions, batch, hidden_dim)
        h0 = torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).to(device=self.device,
                                                                                                    dtype=self.fdtype)
        if(isinstance(self.rnn, nn.LSTM)):
            c0 = torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim
                             ).to(device=self.device, dtype=self.fdtype)
            hiddenvec = (h0, c0)
        else:
            hiddenvec = h0
        return(hiddenvec)

    def _reshape_rnn_output(self, rnn_out):
        """
        Args:
            rnn_out: torch tensor, (batch, seq_len, num_directions * hidden_size)
        """
        encoder_approach = self.config.get('encoder_approach')
        if(encoder_approach == '[h_f+h_b]'):
            return rnn_out[:, :, :self.hidden_dim] + rnn_out[:, :, self.hidden_dim:]
        else:
            return rnn_out

    def forward(self, doc_tensor):
        """ perform forward computation

            Args:
                doc_tensor: torch.Tensor, (1, sents, encoding_dim), dtype=torch.float32
                            currently, it accepts one batch (i.e. one doc at a time due to GPU memory limit)
        """

        # init hidden
        num_sents = doc_tensor.size(0)
        hidden = self.init_hidden(num_sents)
        rnn_out, hidden = self.rnn(doc_tensor, hidden)
        # print('rnn_out before', rnn_out.shape)
        # print("rnn_out", "\n", rnn_out)
        # print("hidden", "\n", hidden)
        # print('rnn_out size', rnn_out.shape)
        rnn_out = self._reshape_rnn_output(rnn_out)
        # print('rnn_out after', rnn_out.shape)
        attn_weights_norm = self.attn_model(rnn_out)  # (1, num_sents)
        # print('attn_weights_norm size', attn_weights_norm.size())
        doc_vec = attn_weights_norm.unsqueeze(1).bmm(rnn_out)  # (docs, 1, num_sents) * (docs, num_sents, embed_dim)
        doc_vec = self.dropout_layer(doc_vec.squeeze(1))  # turning (docs, 1, embed_dim) to (docs, embed_dim)
        return doc_vec, attn_weights_norm


class DocEncoder_MeanPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_hiddenlayers=1, bidirection=False, pdropout=0.1,
                 rnn_class=nn.GRU, nonlinear_func=torch.relu, config={}, to_gpu=True, gpu_index=0):

        super(DocEncoder_MeanPooling, self).__init__()
        self.device = get_device(to_gpu, gpu_index)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # dimension of the hidden vector from rnn
        self.num_hiddenlayers = num_hiddenlayers
        self.pdropout = pdropout
        self.dropout_layer = nn.Dropout(pdropout)

        self.config = config
        # to get options for the attention module
        self.fdtype = self.config.get('fdtype', torch.float32)

        if(bidirection):
            self.num_directions = 2
        else:
            self.num_directions = 1

        # rnn module inserts dropout between layers of rnn except for the output of the last layer!
        if(self.num_hiddenlayers == 1 and self.pdropout > 0):
            rnn_dropout = 0
        else:
            rnn_dropout = self.pdropout

        self.rnn = rnn_class(self.input_dim, self.hidden_dim, num_layers=self.num_hiddenlayers, dropout=rnn_dropout,
                             bidirectional=bidirection, batch_first=True)

        self.nonlinear_func = nonlinear_func

    def init_hidden(self, batch_size):
        """initialize hidden vectors at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """
        # a hidden vector has the shape (num_layers*num_directions, batch, hidden_dim)
        h0 = torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).to(device=self.device,
                                                                                                    dtype=self.fdtype)
        if(isinstance(self.rnn, nn.LSTM)):
            c0 = torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim
                             ).to(device=self.device, dtype=self.fdtype)
            hiddenvec = (h0, c0)
        else:
            hiddenvec = h0
        return(hiddenvec)

    def _reshape_rnn_output(self, rnn_out):
        """
        Args:
            rnn_out: torch tensor, (batch, seq_len, num_directions * hidden_size)
        """
        encoder_approach = self.config.get('encoder_approach')
        if(encoder_approach == '[h_f+h_b]'):
            return rnn_out[:, :, :self.hidden_dim] + rnn_out[:, :, self.hidden_dim:]
        else:
            return rnn_out

    def forward(self, doc_tensor):
        """ perform forward computation

            Args:
                doc_tensor: torch.Tensor, (1, sents, encoding_dim), dtype=torch.float32
                            currently, it accepts one batch (i.e. one doc at a time due to GPU memory limit)
        """

        # init hidden
        num_sents = doc_tensor.size(0)
        hidden = self.init_hidden(num_sents)
        rnn_out, hidden = self.rnn(doc_tensor, hidden)
        # print('rnn_out before', rnn_out.shape)
        # print("rnn_out", "\n", rnn_out)
        # print("hidden", "\n", hidden)
        # print('rnn_out size', rnn_out.shape)
        rnn_out = self._reshape_rnn_output(rnn_out)
        # print('rnn_out after', rnn_out.shape)
        doc_vec = rnn_out.mean(axis=1)  # mean pooling across the sentences (docs, embed_dim)
        doc_vec = self.dropout_layer(doc_vec)
        return doc_vec, None  # placeholder for attn_weights


class DocCategScorer(nn.Module):
    def __init__(self, input_dim, num_labels):

        super(DocCategScorer, self).__init__()
        self.input_dim = input_dim  # dimension of the output from :class:`DocEncoder`
        # TODO: do multiple mappings using Sequential or ModuleList
        self.classifier = nn.Linear(input_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, doc_tensor):
        """ perform forward computation

            Args:
                doc_tensor: torch.Tensor, (1, encoding_dim), dtype=torch.float32
                            currently, it accepts one batch (i.e. one doc at a time due to GPU memory limit)
        """

        # init hidden
        out = self.classifier(doc_tensor)
        # print('classifier ', out)
        # print(out.size())
        # compute log soft max
        out = self.logsoftmax(out)
        # print("classifier out", out.shape)
        return out


def validate_rnn_output(rnn_out, rnn_hidden, config):
    num_directions = config.get('num_directions')
    num_layers = config.get('num_layers')
    h_dim = config.get('hidden_dim')
    batch_size = rnn_out.size(0)
    hn = rnn_hidden.view(num_layers, num_directions, batch_size, h_dim)
    lastlayer_indx = -1
    seqs_len = config.get('seqs_len')
    max_num_elms_inseq = rnn_out.size(1)
    # output has shape (batch_size, max_num_elms_inseq, num_directions, h_dim)
    output = rnn_out.view(batch_size, max_num_elms_inseq, num_directions, h_dim)
    for bindx in range(batch_size):
        for dir_indx in range(num_directions):
            print("batch index: {}, direction index: {}".format(bindx, dir_indx))
            if(dir_indx == 0):  # in case of forward hidden vector, t=T (last time step) should be used
                t = seqs_len[bindx]-1
            else:  # in case of backward hidden vector, t=0 will be the one to use
                t = 0
#             print(hn[lastlayer_indx, dir_indx, bindx, :])
#             print(output[bindx, t, dir_indx, :])
#             print()
            assert torch.equal(hn[lastlayer_indx, dir_indx, bindx, :], output[bindx, t, dir_indx, :])
    print("passed the comparison test!!")


def get_model_numparams(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def restrict_grad_(mparams, mode, limit):
    """clamp/clip a gradient in-place
    """
    if(mode == 'clip_norm'):
        __, maxl = limit
        nn.utils.clip_grad_norm(mparams, maxl, norm_type=2)  # l2 norm clipping
    elif(mode == 'clamp'):  # case of clamping
        minl, maxl = limit
        for param in mparams:
            if param.grad is not None:
                param.grad.data.clamp_(minl, maxl)


def generate_sents_embeds_from_docs(docs_data_tensor, bertembeder, embed_dir, fdtype, gpu_index=0):
    """Generate token embedding for sentences in docs

    Args:
        docs_data_tensor: instance of :class:`DocsDataTensor`
        bertembeder: instance of :class:`BertEmbedder`
        embed_dir: string, path to directory where to dump embedding per document
        fdtype: torch dtype, {torch.float32 or torch.float64}

    """
    bert_proc_docs = {}
    gpu_device = get_device(to_gpu=True, index=gpu_index)
    # move bertembedder to gpu
    bertembeder.type(fdtype).to(gpu_device)
    samples_counter = 0
    num_iter = len(docs_data_tensor)  # number of samples
    for doc_indx in range(num_iter):
        print(doc_indx)
        doc_batch, doc_len, doc_sents_len, doc_attn_mask, doc_labels, doc_id = docs_data_tensor[doc_indx]
        # push to gpu
        embed_sents = bertembeder(doc_batch.to(gpu_device), doc_attn_mask.to(gpu_device), doc_len.item())
        # write to disk for now
        embed_fpath = os.path.join(embed_dir, '{}.pkl'.format(doc_id))
        ReaderWriter.dump_tensor(embed_sents, embed_fpath)
        # add embedding to dict
        bert_proc_docs[doc_id] = embed_fpath
        # clean stuff
        del embed_sents
        # torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()
        samples_counter += 1
        print("processed doc id: {}, {}/{}".format(doc_id, samples_counter, num_iter))
    return bert_proc_docs
