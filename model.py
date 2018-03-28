import torch
import logging
import torch.optim as optim
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F

from rnn_reader import RnnDocReader

logger = logging.getLogger(__name__)


class DocReader(object):
    """High level model that handles initializing the underlying network
    architecture,saving,updating examples, and predicting examples.
    """

    # ----------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------

    def __init__(self, args, word_dict, feature_dict, char_dict,
                 state_dict=None, normalize=True):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.char_dict = char_dict
        self.args.char_size = len(char_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize is false,scores are not nomalized
        # 0-1 per paragraph (no softmax)
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            logger.info("!!!!!!!!! Having state_dict")
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def load_char_embeddings(self, chars, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        chars = {c for c in chars if c in self.char_dict}
        logger.info('Loading pre-trained char embeddings for %d chars from %s' %
                    (len(chars), embedding_file))
        embedding = self.network.char_embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.char_dict.normalize(parsed[0])
                if w in chars:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.char_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate char embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.char_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.char_dict[w]].div_(c)

        logger.info('Loaded %d char embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(chars)))

    def tune_embeddings(self, words):
        """
        Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        :param words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """
        Initialize an optimizer for the free parameters of the network.
        :param state_dict:  network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """
        Forward a batch of examples; step the optimizer to update weights.
        :param ex:
        :return:
        """
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        # Run forward
        score_s, score_e = self.network(*inputs)  # batch_size * doc_max_len

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)  # Negative log Lokelihood

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex[0].size(0)

    def reset_parameters(self):
        """
        Reset any partially fixed parameters to original states.
        :return:
        """
        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.args.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """
        Forward a batch of examples only to get predictions.
        :param ex: the batch
        :param candidates: batch * variable length list of string answer options.
                          The model will only consider exact spans contained in this list.
        :param top_n: Number of predictions to return per batch element.
        :param async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        :return:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Teansfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:5]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """
        Take argmax of constrained score_s * score_e.
        :param score_s: independent start predictions
        :param score_e: independent end predictions
        :param top_n: number of top scored pairs to take
        :param max_len: max span length to consider
        :return:
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])  # 计算两个向量的张量积

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)  # Upper(Lower) triangle of an arrary

            # Take argmax or top n
            scores = scores.numpy()
            score_flat = scores.flatten()  # s copy of the arrary collapsed into one dimension
            if top_n == 1:
                idx_sort = [np.argmax(score_flat)]  # return the indices of the maximum values along an axis
            elif len(score_flat) < top_n:
                idx_sort = np.argsort(-score_flat)  # return the indices that would sort an array
            else:
                idx = np.argpartition(-score_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-score_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(score_flat[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
