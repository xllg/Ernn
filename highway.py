import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class hw(nn.Module):
    """Highway layers
    args:
        size: input and output dimension
        dropout_ratio: dropout ratio
    """

    def __init__(self, size, num_layers=1, dropout_ratio=0.5):
        super(hw, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_ratio)

        for i in range(num_layers):
            tmptrans = nn.Linear(size, size)
            tmpgate = nn.Linear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def rand_init(self):
        """
        random initialization
        """
        for i in range(self.num_layers):
            self.init_linear(self.trans[i])
            self.init_linear(self.gate[i])

    def init_linear(input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def forward(self, x):
        """
        update statics for f1 score

        args:
            x (ins_num, hidden_dim): input tensor
        return:
            output tensor (ins_num, hidden_dim)
        """

        g = F.sigmoid(self.gate[0](x))
        h = F.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = self.dropout(x)
            g = F.sigmoid(self.gate[i](x))
            h = F.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x