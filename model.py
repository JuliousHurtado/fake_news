import torch.nn as nn
import torch
import math
import time
import random

#torch.manual_seed(1234)
#torch.cuda.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)


class Net(nn.Module):
    def __init__(self, n_hidden = 128, vect_dim = 300):
        super(Net, self).__init__()

        self.multiHead = MultiHead()

        self.rnn1 = nn.GRUCell(vect_dim, n_hidden)
        self.rnn2 = nn.GRUCell(vect_dim, n_hidden)

        self.hidden_size = n_hidden
        
        self.fc1 = nn.Linear(n_hidden*2, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):

        x1 = self.multiHead(x1)
        x2 = self.multiHead(x2)

        output1 = self.initHidden()
        output2 = self.initHidden()

        for i in range(x1.size(0)):
            output1 = self.rnn1(x1[i].unsqueeze(0), output1)
        for i in range(x2.size(0)):
            output2 = self.rnn2(x2[i].unsqueeze(0), output2)

        out = torch.cat((output1, output2), 1)

        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))     

        return out

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)

class SelfAttention(nn.Module):
    def __init__(self, vect_dim = 300, n_hidden = 300, p_drop = 0.5):
        super(SelfAttention, self).__init__()

        self.vect_dim = torch.Tensor([vect_dim]).to(device)

        self.q = nn.Linear(vect_dim, n_hidden)
        self.k = nn.Linear(vect_dim, n_hidden)
        self.v = nn.Linear(vect_dim, n_hidden)

        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(p_drop)
        self.bn = nn.LayerNorm(n_hidden)

    def forward(self, x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        out = torch.matmul(q, k.t())
        out = out / torch.sqrt(self.vect_dim)
        out = self.softmax(out)
        out = self.dropout(out)
        out = torch.matmul(out, v)

        out += x

        out = self.bn(out)

        return out

class MultiHead(nn.Module):
    def __init__(self, vect_dim = 300, n_hidden = 300, n_stack = 3, p_drop= 0.5):
        super(MultiHead, self).__init__()

        self.n_stack = n_stack

        self.attention = SelfAttention()

        self.w = nn.Linear(vect_dim, n_hidden)
        self.dropout = nn.Dropout(p_drop)

        self.layer_norm = nn.LayerNorm(vect_dim)

    def forward(self, x):

        output = x
        for i in range(self.n_stack):
            output = self.attention(output)

            output = self.dropout(self.w(output))
            output = self.layer_norm(output + x)

        return output