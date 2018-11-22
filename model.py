import torch.nn as nn
import torch
import math
import time
import random

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

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

        self.rnn1 = nn.GRUCell(vect_dim, n_hidden)
        self.rnn2 = nn.GRUCell(vect_dim, n_hidden)

        self.hidden_size = n_hidden
        
        self.fc1 = nn.Linear(n_hidden*2, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
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
