# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNNClassic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size = None, model="gru", n_layers=1, batch_first=True):
        super(CharRNNClassic, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if embedding_size == None:
            self.embedding_size = hidden_size
        else:
            self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.batch_first = batch_first

        self.encoder = nn.Embedding(input_size, self.embedding_size)
        if self.model == "gru":
            self.rnn = nn.GRU(self.embedding_size, hidden_size, n_layers, batch_first=self.batch_first)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(self.embedding_size, hidden_size, n_layers, batch_first=self.batch_first)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        #batch_size = input.size(0)
        if(self.batch_first is True):
            encoded = self.encoder(input)
            output, hidden = self.rnn(encoded, hidden)
            output = self.decoder(output)
        else:
            batch_size = input.size(0) 
            encoded = self.encoder(input) 
            output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden) 
            output = self.decoder(output.view(batch_size, -1))
        
        
        return output, hidden

    def forward2(self, input, hidden):
        if(self.batch_first is True):
            encoded = self.encoder(input)
            encoded = encoded.unsqueeze(1)  # (1, 1, embed)
            output, hidden = self.rnn(encoded, hidden)
            output = self.decoder(output)
        else:
            encoded = self.encoder(input.view(1, -1)) 
            output, hidden = self.rnn(encoded.view(1, 1, -1), hidden) 
            output = self.decoder(output.view(1, -1)) 
            
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

