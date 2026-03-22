# https://github.com/spro/char-rnn.pytorch for CharRNNClassic

import torch
import torch.nn as nn
import torch.nn.functional as F
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



        

class CharRNNCustom(nn.Module):
    def __init__(self, input_size, output_size, gru_hidden_sizes, lstm_hidden_size, embedding_size, batch_first=True):
        super().__init__()
        self.gru_hidden_sizes = gru_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.batch_first = batch_first

        self.encoder = nn.Embedding(input_size, embedding_size)
        #GRU Representation for character/word/sentence structure/spelling (syntax). At these GRU layers DropConnect
        
        #This Gru Layer would be for character level relations (which characters are most likely to come after others to form words)
        self.chargru = nn.GRU(embedding_size, gru_hidden_sizes[0], 1, batch_first=batch_first)
        self.chardropout = nn.Dropout(p=0.1)
        #This Gru Layer would be for word level relations (which word is most likely to come after the other, noun/verb syntax rules, etc)
        self.wordgru = nn.GRU(gru_hidden_sizes[0], gru_hidden_sizes[1], 1, batch_first=batch_first)
        self.worddropout = nn.Dropout(p=0.2)
         #This Gru Layer would be for sentence level relations (how should each sentence be formed, subject-predicate styled structure and etc)
        self.sentgru = nn.GRU(gru_hidden_sizes[1], gru_hidden_sizes[2], 1, batch_first=batch_first)
        self.sentdropout = nn.Dropout(p=0.3)

        #LTSM Representation for story structure. Dropout of any variety is not applied

        #This LTSM layer should be very wide and takes in the GRU layer's for syntax as input. Attempts to learn story structure
        self.storylstm = nn.LSTM(gru_hidden_sizes[2], lstm_hidden_size, 1, batch_first=batch_first)
        
        
        self.decoder = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, input, hidden):
        char_h, word_h, sent_h, lstm_h = hidden

        encoded = self.encoder(input)
        char_out, char_out_h = self.chargru(encoded, char_h)
        char_out = self.chardropout(char_out)
        word_out, word_out_h = self.wordgru(char_out, word_h)
        word_out = self.worddropout(word_out)
        sent_out, sent_out_h = self.sentgru(word_out, sent_h)
        sent_out = self.sentdropout(sent_out)
        lstm_out, lstm_out_h = self.storylstm(sent_out, lstm_h)
        output = self.decoder(lstm_out)

        return output, (char_out_h, word_out_h, sent_out_h, lstm_out_h)
    
    def forward2(self, input, hidden):
        char_h, word_h, sent_h, lstm_h = hidden

        encoded = self.encoder(input)
        encoded = encoded.unsqueeze(1)  

        encoded = self.encoder(input)
        char_out, char_out_h = self.chargru(encoded, char_h)
        char_out = self.chardropout(char_out)
        word_out, word_out_h = self.wordgru(char_out, word_h)
        word_out = self.worddropout(word_out)
        sent_out, sent_out_h = self.sentgru(word_out, sent_h)
        sent_out = self.sentdropout(sent_out)
        lstm_out, lstm_out_h = self.storylstm(sent_out, lstm_h)
        output = self.decoder(lstm_out)

        return output, (char_out_h, word_out_h, sent_out_h, lstm_out_h)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.gru_hidden_sizes[0], device=device),
            torch.zeros(1, batch_size, self.gru_hidden_sizes[1], device=device),
            torch.zeros(1, batch_size, self.gru_hidden_sizes[2], device=device),
            (torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
             torch.zeros(1, batch_size, self.lstm_hidden_size, device=device))
        )
