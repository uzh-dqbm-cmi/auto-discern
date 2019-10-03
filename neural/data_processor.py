from copy import deepcopy
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .dataset import DocDataTensor


class DataDictProcessor(object):
    def __init__(self, config):
        '''
        Args
            config: dict, specifying options
                - torch_device: instance of torch.device -- ommitting it for now
                - label_cutoff: int, from 1 to 5
                - label_avgmethod: string, {round_mean, floor_mean}
                - tokenizer_max_sent_len: int, in range(1, 512)
        '''
        self.config = config
        self.articles_repr = None
        self.articles_dict = None

    def get_artid_label_map(self, article_id, cutoff=3, method='round_mean'):
        articles_dict = self.articles_dict
        # every sentence has same labels associated with the doc comprising them
        access_key = "{}-{}".format(article_id, 0)
        labels = articles_dict[access_key]['responses'].values.mean(axis=1)
        if(method == 'round_mean'):
            labels = np.round(labels)
        elif(method == 'floor_mean'):
            labels = np.floor(labels)

        mask = labels >= cutoff
        labels[mask] = 1
        labels[~mask] = 0

        return(labels)

    def generate_doctensor_from_articles(self, tokenizer):
        '''Generate tensor representation of the docs in the processed data dictionary 

        Args:
            tokenizer: instance of :class:`BertTokenizer`


        .. Note::

            this function is called after having `self.articles_repr` created using :func:`self.generate_articles_repr` 
            or setting it through :func:`self.set_instance_attr`
        '''
        articles_repr = self.articles_repr
        docs_batch = []
        docs_sents_len = []
        docs_attn_mask = []
        docs_len = []
        articles_id = list(articles_repr.keys())
        # sort for keeping the order when we map the doc_ids to sentences tensor
        articles_id.sort()
        indx_doc_map = {indx: art_id for indx, art_id in enumerate(articles_id)}
        cutoff = self.config.get('label_cutoff', 3)
        avg_method = self.config.get('label_avgmethod', 'round_mean')
        labels = []

        for doc_id in articles_id:
            article_repr = articles_repr[doc_id]
            batched_sents_ids, sents_tok, sents_len, attn_mask = self._generate_doctensor_from_article(article_repr,
                                                                                                       tokenizer)
            docs_batch.append(batched_sents_ids)
            docs_sents_len.append(sents_len)
            docs_attn_mask.append(attn_mask)
            self.articles_repr[doc_id]['sents_tok'] = sents_tok
            # get labels for questions
            responses = self.get_artid_label_map(doc_id, cutoff=cutoff, method=avg_method)
            # turn numpy array into torch.Tensor with 1xC dimension where C is len(labels)
            responses = torch.from_numpy(responses).unsqueeze(0).type(torch.uint8)
            labels.append(responses)
            docs_len.append(article_repr['num_sents'])

        # (docs, num_sents, max_sent_len)
        docs_batch = pad_sequence(docs_batch, batch_first=True, padding_value=0)
        # (docs, num_sents)
        docs_sents_len = pad_sequence(docs_sents_len, batch_first=True, padding_value=0)
        # (docs, num_sents, max_sent_len)
        docs_attn_mask = pad_sequence(docs_attn_mask, batch_first=True, padding_value=0)
        # (docs, num_questions)
        docs_labels = torch.cat(labels, dim=0)
        # (docs, )
        docs_len = torch.tensor(docs_len, dtype=torch.int16)

        return DocDataTensor(docs_batch, docs_len, docs_sents_len, docs_attn_mask, docs_labels, indx_doc_map)
    
    def _generate_doctensor_from_article(self, article_repr, tokenizer):
        '''generate tensor representation of a processed article'''

        sents_tok = []
        sents_len = []
        max_sent_len = self.config.get('tokenizer_max_sent_len', 256)
        if(max_sent_len > 510):  # BERT tokenizer has max len equal to 512 (i.e. number of tokens)
            max_sent_len = 510  # make sure to leave two tokens for the CLS and SEP

        # placeholder when padding sentences with different lengths
        sents_ids = [torch.tensor([0]*(max_sent_len+2), dtype=torch.long)]
        num_sents = article_repr['num_sents']
        attn_mask = torch.zeros((num_sents, max_sent_len+2), dtype=torch.uint8)  # binary mask
        
        for sent_indx, sent in enumerate(article_repr['sents']):
            # sandw_sent = '[CLS] ' + sent + ' [SEP]'
            # toks = tokenizer.tokenize(sandw_sent)

            toks = tokenizer.tokenize(sent)
            # sandwich the sentence with [CLS] and [SEP] symbols
            toks = ['[CLS]'] + toks[:max_sent_len] + ['[SEP]']
            sents_tok.append(toks)
            num_toks = len(toks)
            attn_mask[sent_indx, :num_toks] = 1  # set 1 for tokens that are not padding
            sents_len.append(num_toks)
            toks_ids = tokenizer.convert_tokens_to_ids(toks)
            sents_ids.append(torch.tensor(toks_ids, dtype=torch.long))
            # TODO: intervene here with BertModel to generate embedded representation (i.e. as feature extractor)
        
        # padd sequences to obtain (sents, max_sent_len); sents here refers to number of sents in the article
        # padd sequences to get BxTx* shape
        batched_sents_ids = pad_sequence(sents_ids, batch_first=True, padding_value=0)
        # remove first sentence placeholder
        batched_sents_ids = batched_sents_ids[1:, :]
        # print(batched_sents_ids, '\n', batched_sents_ids.size())

        # tensorize!!
        # (batch_size, max_sent_len)
        sents_len = torch.tensor(sents_len, dtype=torch.int16)  # (num_sents,)
        
        return batched_sents_ids, sents_tok, sents_len, attn_mask

    def set_instance_attr(self, articles_repr, articles_dict, config):
        self.articles_repr = articles_repr
        self.articles_dict = articles_dict
        self.config = config
        
    def generate_articles_repr(self, data_dict):
        articles_repr = {}
        articles_dict = {}
        # get articles id
        articles_id = {int(artsent_id.split('-')[0]) for artsent_id in data_dict}
        for article_id in articles_id:
            article_dict, article_repr = self._generate_article_repr(article_id, data_dict)
            articles_repr.update(article_repr)
            articles_dict.update(article_dict)
        # set the instance variables
        self.articles_repr = articles_repr
        self.articles_dict = articles_dict
    
    def _generate_article_repr(self, article_id, data_dict):
        '''Remove pseudo-sentences (i.e. ones that have only full-stop or one character) and generate new representaiton
         from a parsed article

        Args:
            article_id: int/string, representing article/doc number
            data_dict: dict, pre-processed representation of articles/docs
        
        Returns:
            article_dict: dict, updated/cleaned version data_dict for the specified article id
            article_info: dict, {article_id:{'content': article text,
                                             'sents': list of sentences,
                                             'num_sents': number of sentences}}

        .. Note::
            Period or single character senteneces are generally generated due to having links/images that have no
            content between the tags
        '''
        counter = 0
        article_info = {article_id: {}}
        article_dict = {}
        accum_sent_lst = []
        upd_counter = 0
        # using this loop with exception is in order to sequentially access the article's sentences in the correct order
        while True:
            try:
                sent = data_dict['{}-{}'.format(article_id, counter)]['content']
                if(sent == '.' or len(sent) == 1):
                    print("Removing: ", sent)
                else:
                    dict_key = '{}-{}'.format(article_id, upd_counter)
                    article_dict[dict_key] = deepcopy(data_dict['{}-{}'.format(article_id, counter)])
                    curr_dict = article_dict['{}-{}'.format(article_id, upd_counter)]
                    # update sub_id
                    curr_dict['sub_id'] = upd_counter
                    # update metamap index info
                    if('metamap_detail' in curr_dict):
                        for metamap_dict in curr_dict['metamap_detail']:
                            metamap_dict['index'] = "'{}-{}'".format(article_id, upd_counter)
                    accum_sent_lst.append(sent)
                    upd_counter += 1
                counter += 1
            except Exception:
                print('Finished processing data for article id: ', article_id)
                article_info[article_id]['sents'] = accum_sent_lst
                article_info[article_id]['content'] = " ".join(accum_sent_lst)
                article_info[article_id]['num_sents'] = upd_counter
                break
        return article_dict, article_info
