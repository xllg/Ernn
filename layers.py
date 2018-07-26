import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class StackedBRNN(nn.Module):
    """
    Stacked Bi-directional RNNs.堆叠的双向RNN

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size)
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()  # H(x, W_h)
        self.trans = nn.ModuleList()  # T(x, W_h)
        for i in range(num_layers):
            input_size = input_size if i ==0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))
            self.trans.append(nn.Linear(2 * hidden_size, 2 * hidden_size))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.

        :param x: batch * len * hdim
        :param x_mask: batch * len (1 for padding, 0 for true)
        :return: x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval(评估).
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding"""
        # Transpose batch and sequence dims
        x = x.transpose(0,1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input, p=self.dropout_rate, training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]  # H(x, W_h)
            # if i > 0:  # highway
            #     trans_gate = F.sigmoid(self.trans[i](rnn_input))  # T(x, W_h)
            #     highway_one = torch.mul(rnn_output, trans_gate)  # H(x, W_h) * T(x, W_h)
            #     highway_two = torch.mul(rnn_input, 1 - trans_gate)  # x * (1 - T(x, W_h))
            #     rnn_output = torch.add(highway_one, highway_two)  # H(x, W_h) * T(x, W_h) + x * (1 - T(x, W_h))
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:],2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0,1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)

        return output

    def _forward_padded(self, x, x_mask):
        """Slower (sifnificantly), but more precise, encoding that handles padding."""
        # Compute sorted sequence lengths
        # eq():比较两个元素是否相等，相等1，否则0
        # long():将Tensor转换成long类型
        # sum(input,dim,out=None) 返回输入张量指定维度上每行的和
        # squeeze():将输入张量中维度为1去除
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        # sort:对输入张量沿着指定的维度按升序排序，若不给定dim，则默认为输入的最后一维，descending为True表示降序排序,返回排序后的张量和下标
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                      p=self.dropout_rate,
                                      training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            rnn_output = self.rnns[i](rnn_input)[0]  # H(x, W_h)
            # if i > 0:  # highway
            #     trans_gate = F.sigmoid(self.trans[i](rnn_input[0]))  # T(x, W_h)
            #     highway_one = torch.mul(rnn_output[0], trans_gate)  # H(x, W_h) * T(x, W_h)
            #     highway_two = torch.mul(rnn_input[0], 1 - trans_gate)  # x * (1 - T(x, W_h))
            #     rnn_output[0].data = torch.add(highway_one, highway_two).data  # H(x, W_h) * T(x, W_h) + x * (1 - T(x, W_h))
            outputs.append(rnn_output)

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y,match sequence Y to each element in X
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self,input_size,identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size,input_size)
        else:
            self.linear = None

    def forward(self,x,y,y_mask):
        """
        :param x: batch * len1 * hdim
        :param y: batch * len2 * hdim
        :param y_mask: batch * len2 (1 for padding,0 for true)
        :return: matched_seq:batch * len1 * hdim
        """

        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        # unsqueeze:return a new tensor with a dimension of size one inserted at the specified position
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        # data.masked_fill_(mask,value):在mask值为1的位置处用value填充data
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.cdfl = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, dropout=self.dropout)
        self.cdbl = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, dropout=self.dropout)
        self.cqfl = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, dropout=self.dropout)
        self.cqbl = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, dropout=self.dropout)

        # self.docf2char = hw(self.hidden_size)
        # self.docb2char = hw(self.hidden_size)
        # self.qesf2char = hw(self.hidden_size)
        # self.qesb2char = hw(self.hidden_size)

        self.doc_init = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.qes_init = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, d_f, d_f_p, d_b, d_b_p, q_f, q_f_p, q_b, q_b_p):
        # lstm计算字符向量
        d_f_lo, _ = self.cdfl(d_f)
        d_b_lo, _ = self.cdbl(d_b)
        q_f_lo, _ = self.cqfl(q_f)
        q_b_lo, _ = self.cqbl(q_b)

        # highway network
        # d_f_lo = self.docf2char(d_f_lo, d_f)
        # d_b_lo = self.docb2char(d_b_lo, d_b)
        # q_f_lo = self.qesf2char(q_f_lo, q_f)
        # q_b_lo = self.qesb2char(q_b_lo, q_b)

        # 根据每个结尾字符的位置，取得隐藏层状态
        # -------------文章----------------
        d_f_p = d_f_p.unsqueeze(2).expand(d_f_p.size(0), d_f_p.size(1), d_f_lo.size(2))
        d_f_g = torch.gather(d_f_lo, 1, d_f_p)
        d_b_p = d_b_p.unsqueeze(2).expand(d_b_p.size(0), d_b_p.size(1), d_b_lo.size(2))
        d_b_g = torch.gather(d_b_lo, 1, d_b_p)

        # -------------问题----------------
        q_f_p = q_f_p.unsqueeze(2).expand(q_f_p.size(0), q_f_p.size(1), q_f_lo.size(2))
        q_f_g = torch.gather(q_f_lo, 1, q_f_p)
        q_b_p = q_b_p.unsqueeze(2).expand(q_b_p.size(0), q_b_p.size(1), q_b_lo.size(2))
        q_b_g = torch.gather(q_b_lo, 1, q_b_p)

        # 将反向结果进行倒序
        d_b_g = trilone(d_b_g)
        q_b_g = trilone(q_b_g)

        # 将前向和反向编码通过linear整合
        doc_char_emb = F.dropout(F.relu(self.doc_init(torch.cat([d_f_g, d_b_g], 2))), p=self.dropout)
        qes_char_emb = F.dropout(F.relu(self.qes_init(torch.cat([q_f_g, q_b_g], 2))), p=self.dropout)

        return doc_char_emb, qes_char_emb

class hw(nn.Module):
    def __init__(self, size):
        super(hw, self).__init__()
        self.trans = nn.Linear(size, size)

    def forward(self, h_x, x):
        t = F.sigmoid(self.trans(x))
        x = h_x * t + (1 - t) * x
        return x

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------

def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def trilone(x_proj):
    tril_one = torch.ones(x_proj.size(1), x_proj.size(1)).tril().triu()
    tril_one = tril_one.tolist()
    tril_one.reverse()
    tril_one = torch.Tensor(tril_one)
    tril = torch.Tensor(x_proj.size(0), x_proj.size(1), x_proj.size(1)).copy_(tril_one)
    tril = Variable(tril).cuda()
    x_proj = torch.matmul(tril, x_proj)
    return x_proj










