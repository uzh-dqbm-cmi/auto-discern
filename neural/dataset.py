import os
import numpy as np
import torch
from .utilities import ModelScore
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


class DocDataTensor(Dataset):
    
    def __init__(self, docs_batch, docs_len, docs_sents_len, docs_attn_mask, docs_labels, indx_doc_map):
        self.docs_batch = docs_batch  # tensor.long, (docs, num_sents, sents_len)
        self.docs_len = docs_len  # tensor.uint8 (docs,), number of sentences in each doc
        self.docs_sents_len = docs_sents_len  # tensor.uint8, (docs, num_sents)
        self.docs_attn_mask = docs_attn_mask  # tensor.uint8, (docs, num_sents, sents_len)
        self.docs_labels = docs_labels  # tensor.uint8, (docs, num_questions)
        self.indx_doc_map = indx_doc_map  # dict, {indx:doc_id}
        self.doc_indx_map = {doc_id: indx for indx, doc_id in self.indx_doc_map.items()}  # dict, indx_doc_map reversed
        self.num_samples = docs_batch.size(0)  # int, number of docs
        
    def __getitem__(self, indx):
        
        return(self.docs_batch[indx], self.docs_len[indx], 
               self.docs_sents_len[indx], self.docs_attn_mask[indx],
               self.docs_labels[indx], self.indx_doc_map[indx])
  
    def __len__(self):
        return(self.num_samples)
    
    
class PartitionDataTensor(Dataset):
    
    def __init__(self, doc_data_tensor, partition_ids, dsettype, fold_num):
        self.docs_data_tensor = doc_data_tensor  # instance of :class:`DocDataTensor`
        self.partition_ids = partition_ids  # list of doc ids
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.fold_num = fold_num  # int, fold number
        self.num_samples = len(self.partition_ids)  # int, number of docs in the partition TODO: update this part
        
    def __getitem__(self, indx):
        doc_id = self.partition_ids[indx]
        upd_indx = self.docs_data_tensor.doc_indx_map[doc_id]
        return self.docs_data_tensor[upd_indx]
  
    def __len__(self):
        return(self.num_samples)


def construct_load_dataloaders(dataset_fold, dsettypes, config, wrk_dir):
    """construct dataloaders for the dataset

       Args:
            dataset_fold: dictionary,
                          example: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                                    'class_weights': tensor([0.6957, 1.7778])
                                   }
            dsettype: list, ['train', 'validation', 'test']
            config: dict, {'batch_size': int, 'num_workers': int}
            wrk_dir: string, folder path
    """

    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    epoch_loss_avgsamples = {}
    flog_out = {}
    score_dict = {}
    class_weights = {}
    for dsettype in dsettypes:
        if(dsettype == 'train' or dsettype == 'validation'):
            shuffle = True
            class_weights[dsettype] = dataset_fold['class_weights']
        else:
            shuffle = False
            class_weights[dsettype] = None
        data_loaders[dsettype] = DataLoader(dataset_fold[dsettype],
                                            batch_size=config['batch_size'],
                                            shuffle=False,  # TODO: update this
                                            num_workers=config['num_workers'])

        epoch_loss_avgbatch[dsettype] = []
        epoch_loss_avgsamples[dsettype] = []
        score_dict[dsettype] = ModelScore(0, 0.0, 0.0, 0.0, 0.0)  # (best_epoch, micro_f1, macro_f1, accuracy, auc)
        if(wrk_dir):
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
        else:
            flog_out[dsettype] = None

    return (data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out)


def get_stratified_partitions(docs_data_tensor, questions=(4, 5, 9, 10, 11), num_folds=5, valid_set_portion=0.1, random_state=42):
    """Generate 5-fold stratified sample of document ids based on the question label
    
    Args:
        docs_data_tensor: instance of :class:`DocDataTensor`
    
    """
    skf_trte = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)  # split train and test
    sss_trv = StratifiedShuffleSplit(n_splits=1, random_state=random_state, test_size=valid_set_portion)  # split train and validation
    docs_labels = docs_data_tensor.docs_labels  # tensor (docs, num_questions)
    get_docs_id = np.vectorize(docs_data_tensor.indx_doc_map.get)  # vectorized lookup
    q_partitions = {}
    for question in questions:
        print("question", question)
        q_partitions[question] = {}
        question_indx = question - 1  # indexing starts from 0
        q_labels = docs_labels[:, question_indx].numpy()  # all labels
        x = np.zeros(len(q_labels))   # placeholder for compatibility reasons
        fold_count = 0
        for train_index, test_index in skf_trte.split(x, q_labels):
            # further split train_set to 90% train and 10% validation
            q_train_labels = q_labels[train_index]
            x_train = np.zeros(len(q_train_labels))
            train_doc_ids = get_docs_id(train_index)
            test_doc_ids = get_docs_id(test_index)
            
            for tr_index, validation_index in sss_trv.split(x_train, q_train_labels):  # loop runs once
                tr_doc_ids = train_doc_ids[list(tr_index)]
                val_doc_ids = train_doc_ids[list(validation_index)]
                q_partitions[question][fold_count] = {'train': tr_doc_ids,
                                                      'validation': val_doc_ids,
                                                      'test': test_doc_ids}
            print("fold_num:", fold_count)
            print('train data')
            report_label_distrib(q_train_labels[tr_index])
            print('validation data')
            report_label_distrib(q_train_labels[validation_index])
            print('test data')
            report_label_distrib(q_labels[test_index])
            print()
            fold_count += 1
        print("-"*25)
    return(q_partitions)


def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])


def validate_partitions(q_partitions, docs_id, valid_set_portion=0.1, test_set_portion=0.2, num_docs=269):
    if(not isinstance(docs_id, set)):
        docs_id = set(docs_id)
    print("validation partitions")
    for question in q_partitions:
        test_set_accum = set([])
        print('question', question)
        for fold_num in q_partitions[question]:
            print('fold_num', fold_num)
            tr_ids = q_partitions[question][fold_num]['train']
            val_ids = q_partitions[question][fold_num]['validation']
            te_ids = q_partitions[question][fold_num]['test']

            tr_val = set(tr_ids).intersection(val_ids)
            tr_te = set(tr_ids).intersection(te_ids)
            te_val = set(te_ids).intersection(val_ids)
            
            tr_size = len(tr_ids) + len(val_ids)
            num_docs = tr_size + len(te_ids)
            print('expected validation set size:', valid_set_portion*tr_size, '; actual validation set size:', len(val_ids))
            print('expected test set size:', test_set_portion*num_docs, '; actual test set size:', len(te_ids))
            print()
            assert np.abs(valid_set_portion*tr_size - len(val_ids)) <= 2  # valid difference range
            assert np.abs(test_set_portion*num_docs - len(te_ids)) <= 2
            # assert there is no overlap among train, val and test partition within a fold
            for s in (tr_val, tr_te, te_val):
                assert len(s) == 0

            s_union = set(tr_ids).union(val_ids).union(te_ids)
            assert len(s_union) == num_docs
            test_set_accum = test_set_accum.union(te_ids)
        print('-'*25)
        # verify that assembling test sets from each of the five folds would be equivalent to all doc_ids
        assert len(test_set_accum) == num_docs
        assert test_set_accum == docs_id
    print("passed intersection and overlap test (i.e. train, validation and test sets are not",
          "intersecting in each fold and the concatenation of test sets from each fold is",
          "equivalent to the whole dataset)")


def validate_doc_tensor_repr(doc_id, doc_data_tensor, processor, tokenizer):
    docs_batch = doc_data_tensor.docs_batch
    doc_indx = doc_data_tensor.doc_indx_map[doc_id]
    num_sents = doc_data_tensor.docs_len[doc_indx]
    print('num_sents:', num_sents)
    print('doc_id:', doc_id)
    assert num_sents == processor.articles_repr[doc_id]['num_sents']
    pass_flag = False  # record if we enter the loop
    for sent_indx in range(num_sents):
        sent_len = doc_data_tensor.docs_sents_len[doc_indx, sent_indx]
        a = tokenizer.convert_ids_to_tokens(list(docs_batch[doc_indx, sent_indx, :sent_len].numpy()))
        b = processor.articles_repr[doc_id]['sents_tok'][sent_indx]
        assert a == b
        # print(a)
        # print(b)
        pass_flag = True

    if(pass_flag):
        print("passed!")
    else:
        print("failed!")
    print()


def generate_docpartition_per_question(doc_data_tensor, q_partitions, question):
    q_docpartitions = {question: {}}
    target_q_partitions = q_partitions[question]
    for fold_num in target_q_partitions:
        q_docpartitions[question][fold_num] = {}
        for dsettype in target_q_partitions[fold_num]:
            target_ids = target_q_partitions[fold_num][dsettype]
            doc_partition = PartitionDataTensor(doc_data_tensor, target_ids, dsettype, fold_num)
            q_docpartitions[question][fold_num][dsettype] = doc_partition
    return(q_docpartitions)


def validate_q_docpartitions(q_docpartitions, q_partitions):
    for question in q_docpartitions:
        print('question ', question)
        for fold_num in q_docpartitions[question]:
            print('fold_num', fold_num)
            for dsettype in q_docpartitions[question][fold_num]:
                pdtensor = q_docpartitions[question][fold_num][dsettype]  # partition data tensor instance
                print(id(pdtensor.docs_data_tensor))
                assert np.all(np.equal(np.array(pdtensor.partition_ids), 
                                       np.array(q_partitions[question][fold_num][dsettype])))
    print("passed test!!")


def compute_class_weights(labels_tensor):
    classes, counts = np.unique(labels_tensor, return_counts=True)
    print("classes", classes)
    print("counts", counts)
    class_weights = compute_class_weight('balanced', classes, labels_tensor.numpy())
    return class_weights


def compute_class_weights_per_fold_(q_docpartitions):
    """computes inverse class weights and updates the passed dictionary

    Args:
        q_docpartitions: dictionary, {question, int: {fold_num, int: {datasettype, string:{qdocpartition, instance of :class:`PartitionDataTensor`}}}}
    
    Example:
        q_docpartitions
            {4: {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>
                    }, ..
            }
        is updated after computation of class weights to
            {4: {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                    'class_weights': tensor([0.6957, 1.7778]),
                    }, ..
            }
    """
    labels_pos = 4  # position of labels in the returned tuple when indexing PartitionDataTensor
    for q in q_docpartitions:
        q_indx = q-1
        for fold_num in q_docpartitions[q]:  # looping over the numbered folds
            pdoc = q_docpartitions[q][fold_num]['train']
            q_scores = torch.empty(len(pdoc), dtype=torch.float32).fill_(-1.0)
            for i in range(len(pdoc)):
                out = pdoc[i][labels_pos][q_indx]
                q_scores[i] = out.item()
            q_docpartitions[q][fold_num]['class_weights'] = torch.from_numpy(compute_class_weights(q_scores)).float()
