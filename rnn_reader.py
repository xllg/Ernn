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

        self.char_embedding_dim = 128

        # Word embedding (+1 forpadding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Char embedding (+1 forpadding)
        self.char_embedding = nn.Embedding(args.char_size, self.char_embedding_dim, padding_idx=0)

        # Char encoder
        char_lstm_hidden_size = args.hidden_size
        self.char_doc_forw_lstm = nn.LSTM(self.char_embedding_dim, char_lstm_hidden_size, num_layers=2, bidirectional=False,
                                          dropout=args.dropout_rnn)
        self.char_doc_back_lstm = nn.LSTM(self.char_embedding_dim, char_lstm_hidden_size, num_layers=2, bidirectional=False,
                                          dropout=args.dropout_rnn)
        self.char_qes_forw_lstm = nn.LSTM(self.char_embedding_dim, char_lstm_hidden_size, num_layers=2, bidirectional=False,
                                          dropout=args.dropout_rnn)
        self.char_qes_back_lstm = nn.LSTM(self.char_embedding_dim, char_lstm_hidden_size, num_layers=2, bidirectional=False,
                                          dropout=args.dropout_rnn)

        self.doc_init = nn.Sequential(
            nn.Linear(char_lstm_hidden_size * 2, args.hidden_size),
            nn.ReLU()
        )
        self.qes_init = nn.Sequential(
            nn.Linear(char_lstm_hidden_size * 2, args.hidden_size),
            nn.ReLU()
        )
        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to DocRNN: word emb + question emb +manual features + char emb
        doc_input_size = args.embedding_dim + args.num_features + args.hidden_size
        # doc_input_size = args.embedding_dim + args.hidden_size
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # Input size to QesRNN: question emb + char emb
        qes_input_size = args.embedding_dim + args.hidden_size

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
            input_size=qes_input_size,
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
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

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

    # x1, x1_f, x1_mask, x1_char, x1_char_pos, x2, x2_mask, x2_char, x2_char_pos
    # x1, x1_f, x1_mask, x1_char_forw, x1_char_pos_forw, x1_char_back, x1_char_pos_back,
    # x2, x2_mask, x2_char_forw, x2_char_pos_forw, x2_char_back, x2_char_pos_back, y_s, y_e, ids

    def forward(self, x1, x1_f, x1_mask, x1_char_forw, x1_char_pos_forw, x1_char_back, x1_char_pos_back,
                x2, x2_mask, x2_char_forw, x2_char_pos_forw, x2_char_back, x2_char_pos_back):
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

        x1_char_forw_emb = self.char_embedding(x1_char_forw)
        x2_char_forw_emb = self.char_embedding(x2_char_forw)
        x1_char_back_emb = self.char_embedding(x1_char_back)
        x2_char_back_emb = self.char_embedding(x2_char_back)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x1_char_forw_emb = nn.functional.dropout(x1_char_forw_emb, p=self.args.dropout_emb,
                                                training=self.training)
            x2_char_forw_emb = nn.functional.dropout(x2_char_forw_emb, p=self.args.dropout_emb,
                                                training=self.training)
            x1_char_back_emb = nn.functional.dropout(x1_char_back_emb, p=self.args.dropout_emb,
                                                     training=self.training)
            x2_char_back_emb = nn.functional.dropout(x2_char_back_emb, p=self.args.dropout_emb,
                                                     training=self.training)

        x1_char_forw_emb, _ = self.char_doc_forw_lstm(x1_char_forw_emb)
        x2_char_forw_emb, _ = self.char_qes_forw_lstm(x2_char_forw_emb)
        x1_char_back_emb, _ = self.char_doc_back_lstm(x1_char_back_emb)
        x2_char_back_emb, _ = self.char_qes_back_lstm(x2_char_back_emb)

        x1_char_pos_forw = x1_char_pos_forw.unsqueeze(2).expand(x1_char_pos_forw.size(0), x1_char_pos_forw.size(1), x1_char_forw_emb.size(2))
        x1_char_forw_emb = torch.gather(x1_char_forw_emb, 1, x1_char_pos_forw)
        x1_char_pos_back = x1_char_pos_back.unsqueeze(2).expand(x1_char_pos_back.size(0), x1_char_pos_back.size(1), x1_char_back_emb.size(2))
        x1_char_back_emb = torch.gather(x1_char_back_emb, 1, x1_char_pos_back)

        x2_char_pos_forw = x2_char_pos_forw.unsqueeze(2).expand(x2_char_pos_forw.size(0), x2_char_pos_forw.size(1), x2_char_forw_emb.size(2))
        x2_char_forw_emb = torch.gather(x2_char_forw_emb, 1, x2_char_pos_forw)
        x2_char_pos_back = x2_char_pos_back.unsqueeze(2).expand(x2_char_pos_back.size(0), x2_char_pos_back.size(1), x2_char_back_emb.size(2))
        x2_char_back_emb = torch.gather(x2_char_back_emb, 1, x2_char_pos_back)

        # x1_char_emb = torch.cat([x1_char_forw_emb, x1_char_back_emb], 2)
        # x2_char_emb = torch.cat([x2_char_forw_emb, x2_char_back_emb], 2)
        x1_char_emb = self.doc_init(torch.cat([x1_char_forw_emb, x1_char_back_emb], 2))
        x2_char_emb = self.qes_init(torch.cat([x2_char_forw_emb, x2_char_back_emb], 2))

        x1_char_emb = nn.functional.dropout(x1_char_emb, p=self.args.dropout_emb,
                                                 training=self.training)
        x2_char_emb = nn.functional.dropout(x2_char_emb, p=self.args.dropout_emb,
                                                 training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]
        drnn_input.append(x1_char_emb)

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)
        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(torch.cat([x2_emb, x2_char_emb], 2), x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores

