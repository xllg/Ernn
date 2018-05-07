"""
Functions for putting examples into torch format.
"""

from collections import Counter
import torch

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
    char_doc = torch.zeros(len(ex['document']), args.max_clen)
    char_qes = torch.zeros(len(ex['question']), args.max_clen)
    for i, w in enumerate(ex['document']):
        for j, c in enumerate(w):
            char_doc[i][j] = char_dict[c]
    for i, w in enumerate(ex['question']):
        for j, c in enumerate(w):
            char_qes[i][j] = char_dict[c]

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
        return document, features, question, char_doc, char_qes, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert (len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, char_doc, char_qes, start, end, ex['id']


def batchify(batch):
    """
    Gather a batch of individual examples into one batch
    :param batch:
    :return:
    """
    NUM_INPUTS = 5
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    char_docs = [ex[3] for ex in batch]
    char_questions = [ex[4] for ex in batch]

    # Batch documents, features and char_docs
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    #x1_char = torch.zeros(len(docs), max_length, char_docs[0].size(1))
    x1_char = torch.LongTensor(len(docs), max_length, char_docs[0].size(1)).zero_()
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])
        x1_char[i, :d.size(0)].copy_(char_docs[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    x2_char = torch.LongTensor(len(questions), max_length, char_questions[0].size(1)).zero_()
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
        x2_char[i, :q.size(0)].copy_(char_questions[i])

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x1_char, x2, x2_mask, x2_char, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][5]):
            y_s = torch.cat([ex[5] for ex in batch])
            y_e = torch.cat([ex[6] for ex in batch])
        else:
            y_s = [ex[5] for ex in batch]
            y_e = [ex[6] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x1_char, x2, x2_mask, x2_char, y_s, y_e, ids