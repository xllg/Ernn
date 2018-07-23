"""
Functions for putting examples into torch format.
"""

from collections import Counter
import torch

import itertools

from functools import reduce

def vectorize(ex, model, single_answer=False):
    """Torchify a single example"""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict
    char_dict = model.char_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # Index character
    char_doc_fea = [list(map(lambda t: char_dict[t], w)) for w in ex['document']]
    char_qes_fea = [list(map(lambda t: char_dict[t], w)) for w in ex['question']]

    # document前向字符编码
    doc_word_forw_len = list(map(lambda t: len(t) + 1, char_doc_fea))
    doc_forw_corpus = [char_dict['<NULL>']] + list(reduce(lambda x, y: x + [char_dict['<NULL>']] + y, char_doc_fea)) + [char_dict['<NULL>']]
    char_doc_forw_pos = torch.LongTensor(list(itertools.accumulate(doc_word_forw_len)))
    char_doc_forw = torch.LongTensor(doc_forw_corpus)
    # document反向字符编码
    doc_back_corpus = doc_forw_corpus[::-1]  # 倒序
    doc_word_back_len = doc_word_forw_len[::-1]
    char_doc_back_pos = torch.LongTensor(list(itertools.accumulate(doc_word_back_len)))
    char_doc_back = torch.LongTensor(doc_back_corpus)

    # question前向字符编码
    qes_word_forw_len = list(map(lambda t: len(t) + 1, char_qes_fea))
    qes_forw_corpus = [char_dict['<NULL>']] + list(reduce(lambda x, y: x + [char_dict['<NULL>']] + y, char_qes_fea)) + [char_dict['<NULL>']]
    char_qes_forw_pos = torch.LongTensor(list(itertools.accumulate(qes_word_forw_len)))
    char_qes_forw = torch.LongTensor(qes_forw_corpus)
    # question前向字符编码
    qes_back_corpus = qes_forw_corpus[::-1]  # 倒序
    qes_word_back_len = qes_word_forw_len[::-1]
    char_qes_back_pos = torch.LongTensor(list(itertools.accumulate(qes_word_back_len)))
    char_qes_back = torch.LongTensor(qes_back_corpus)

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, char_doc_forw, char_doc_forw_pos, char_qes_forw, char_qes_forw_pos, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert (len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, char_doc_forw, char_doc_forw_pos, char_doc_back, char_doc_back_pos, \
           char_qes_forw, char_qes_forw_pos, char_qes_back, char_qes_back_pos, start, end, ex['id']


def batchify(batch):
    """
    Gather a batch of individual examples into one batch
    :param batch:
    :return:
    """
    NUM_INPUTS = 11
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    char_docs_forw = [ex[3] for ex in batch]
    char_docs_forw_pos = [ex[4] for ex in batch]
    char_docs_back = [ex[5] for ex in batch]
    char_docs_back_pos = [ex[6] for ex in batch]
    char_qes_forw = [ex[7] for ex in batch]
    char_qes_forw_pos = [ex[8] for ex in batch]
    char_qes_back = [ex[9] for ex in batch]
    char_qes_back_pos = [ex[10] for ex in batch]

    # Batch documents, features and char_docs
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))

    max_cdlen = max([cd.size(0) for cd in char_docs_forw])
    x1_char_forw = torch.LongTensor(len(docs), max_cdlen).zero_()
    x1_char_back = torch.LongTensor(len(docs), max_cdlen).zero_()
    max_cdplen = max([cd.size(0) for cd in char_docs_forw_pos])
    x1_char_pos_forw = torch.LongTensor(len(docs), max_cdplen).zero_()
    x1_char_pos_back = torch.LongTensor(len(docs), max_cdplen).zero_()

    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])
        x1_char_forw[i, :char_docs_forw[i].size(0)].copy_(char_docs_forw[i])
        x1_char_pos_forw[i, :char_docs_forw_pos[i].size(0)].copy_(char_docs_forw_pos[i])
        x1_char_back[i, :char_docs_back[i].size(0)].copy_(char_docs_back[i])
        x1_char_pos_back[i, :char_docs_back_pos[i].size(0)].copy_(char_docs_back_pos[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)

    max_cqlen = max([cd.size(0) for cd in char_qes_forw])
    x2_char_forw = torch.LongTensor(len(questions), max_cqlen).zero_()
    x2_char_back = torch.LongTensor(len(questions), max_cqlen).zero_()
    max_cqplen = max([cd.size(0) for cd in char_qes_forw_pos])
    x2_char_pos_forw = torch.LongTensor(len(questions), max_cqplen).zero_()
    x2_char_pos_back = torch.LongTensor(len(questions), max_cqplen).zero_()

    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
        x2_char_forw[i, :char_qes_forw[i].size(0)].copy_(char_qes_forw[i])
        x2_char_pos_forw[i, :char_qes_forw_pos[i].size(0)].copy_(char_qes_forw_pos[i])
        x2_char_back[i, :char_qes_back[i].size(0)].copy_(char_qes_back[i])
        x2_char_pos_back[i, :char_qes_back_pos[i].size(0)].copy_(char_qes_back_pos[i])

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x1_char_forw, x1_char_pos_forw, x1_char_back, x1_char_pos_back, x2, x2_mask, x2_char_forw, x2_char_pos_forw, x2_char_back, x2_char_pos_back, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][11]):
            y_s = torch.cat([ex[11] for ex in batch])
            y_e = torch.cat([ex[12] for ex in batch])
        else:
            y_s = [ex[11] for ex in batch]
            y_e = [ex[12] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x1_char_forw, x1_char_pos_forw, x1_char_back, x1_char_pos_back, x2, x2_mask, x2_char_forw, x2_char_pos_forw, x2_char_back, x2_char_pos_back, y_s, y_e, ids