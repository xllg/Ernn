""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

import os


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, al):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            # text = context.split(' ')
            # if len(text) == al:
            for qa in paragraph['qas']:
                # ans = qa['answers'][0]['text'].split(' ')  # ans_len
                # qes = qa['question'].split(' ')  # qes_len
                # if len(qes) == al:
                text = qa["question"].split(' ')
                qes_start = text[0].lower()
                if qes_start == al:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'answer_len': al, 'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('--dataset_file', help='Dataset file', default='/home/xllg/PycharmProjects/AttReader/data/datasets/SQuAD-v1.1-dev.json')
    parser.add_argument('--prediction_file', help='Prediction File', default='/home/xllg/PycharmProjects/AttReader/data/drqa_predict')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    ans_len = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29]
    qes_len = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
               31, 33]
    pas_len = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 540, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48,
               49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
               75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
               101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
               122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
               143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
               164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
               185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
               206, 207, 208, 209, 210, 211, 212, 213, 215, 217, 218, 219, 221, 222, 223, 224, 226, 227, 228, 229, 230,
               231, 233, 236, 237, 240, 241, 243, 244, 245, 246, 247, 248, 249, 252, 253, 256, 257, 262, 263, 264, 266,
               267, 274, 276, 277, 278, 279, 283, 286, 293, 294, 296, 297, 302, 304, 312, 322, 324, 327, 333, 337, 340,
               349, 351, 353, 356, 359, 363, 388, 414, 448, 457, 629, 481, 483, 508, 509]
    qes_type = ["why", "where", "which", "when", "who", "how", "what"]
    for al in qes_type:
        outfile = os.path.join(args.prediction_file, 'qes_type_' + al + '.preds')
        with open(outfile) as prediction_file:
            predictions = json.load(prediction_file)
        print(json.dumps(evaluate(dataset, predictions, al)))
