"""Implementation of the RNN based ErnnReader"""

import torch
import torch.nn as nn
from Ernn import layers
# ------------------------------------------------------------
# Network
# ------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize):
        super(RnnDocReader,self).__init__()
        # Store config
        self.args = args

        # Word embedding (+1 forpadding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.seq_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb +manual features + char emb
        doc_input_size = args.embedding_dim + args.num_features + args.char_size
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim * 2,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.qes_self_attn = layers.LinearSeqAttn(question_hidden_size)

        self.qes_gated = layers.LinearGated(doc_hidden_size)
        self.gated_qes_rnn = layers.StackedBRNN(
            input_size=doc_hidden_size * 2,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,  # 3
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        self.doc_gated = layers.LinearGated(question_hidden_size)
        self.gated_doc_rnn = layers.StackedBRNN(
            input_size=doc_hidden_size * 2,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,  # 3
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_mask, x1_char, x2, x2_mask, x2_char):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]
        drnn_input.append(x1_char)

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x1_weighted_emb = self.seq_match(x1_emb, x2_emb, x2_mask) # P2Q
            x2_weighted_emb = self.seq_match(x2_emb, x1_emb, x1_mask) # Q2P
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(torch.cat([x2_emb, x1_weighted_emb], 2), x2_mask)

        gated_q2p_ct = self.qes_gated(doc_hiddens, question_hiddens, x2_mask)
        gated_q2p_vp = self.gated_qes_rnn(gated_q2p_ct, x1_mask)

        gated_p2q_ct = self.doc_gated(question_hiddens, doc_hiddens, x1_mask)
        gated_p2q_vp = self.gated_doc_rnn(gated_p2q_ct, x2_mask)

        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.qes_self_attn(gated_p2q_vp, x2_mask)
        question_hidden = layers.weighted_avg(gated_p2q_vp, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(gated_q2p_vp, question_hidden, x1_mask)
        end_scores = self.end_attn(gated_q2p_vp, question_hidden, x1_mask)
        return start_scores, end_scores