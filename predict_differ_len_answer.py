#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import time
import torch
import argparse
import logging
import json

from tqdm import tqdm
from predictor import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='/home/xllg/PycharmProjects/AttReader/data/datasets/SQuAD-v1.1-dev.json',
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='/home/xllg/PycharmProjects/AttReader/data/drqa_up_predict',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true', default=True,
                    help='Only store single top span instead of top N list')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    args.tokenizer,
    args.embedding_file,
    args.num_workers,
)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------

# predict different length of answer
ans_len = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29]
qes_len = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33]
pas_len = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 540, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 217, 218, 219, 221, 222, 223, 224, 226, 227, 228, 229, 230, 231, 233, 236, 237, 240, 241, 243, 244, 245, 246, 247, 248, 249, 252, 253, 256, 257, 262, 263, 264, 266, 267, 274, 276, 277, 278, 279, 283, 286, 293, 294, 296, 297, 302, 304, 312, 322, 324, 327, 333, 337, 340, 349, 351, 353, 356, 359, 363, 388, 414, 448, 457, 629, 481, 483, 508, 509]

examples_all = [[] for i in range(len(pas_len))] # differ qes ans
qids = [[] for i in range(len(pas_len))] # differ qes ans

with open(args.dataset) as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            text = context.split(' ')
            id = pas_len.index(len(text))
            for qa in paragraph['qas']:
                qids[id].append(qa['id'])
                examples_all[id].append((context, qa['question']))

for num, examples in enumerate(examples_all):
    results = {}
    qid = qids[num]
    for i in tqdm(range(0, len(examples), args.batch_size)):
        predictions = predictor.predict_batch(
            examples[i:i + args.batch_size], top_n=args.top_n
        )
        for j in range(len(predictions)):
            # Official eval expects just a qid --> span
            if args.official:
                results[qid[i + j]] = predictions[j][0][0]

            # Otherwise we store top N and scores for debugging.
            else:
                results[qid[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]

    # model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
    # basename = os.path.splitext(os.path.basename(args.dataset))[0]
    # outfile = os.path.join(args.out_dir, basename + '-' + model + '.preds')
    outfile = os.path.join(args.out_dir, 'pas-len'+str(pas_len[num])+'.preds')

    logger.info('Writing results to %s' % outfile)
    with open(outfile, 'w') as f:
        json.dump(results, f)

    logger.info('Total time: %.2f' % (time.time() - t0))
