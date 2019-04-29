import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import random

use_cuda = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(Encoder, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.enc_lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=HIDDEN_LAYER,            # number of LSTM layer
            batch_first=True,        # (batch, time_step, input_size)
            bidirectional=BIDIR
        )

    def forward(self, input, hidden):
        output, hidden = self.enc_lstm(input, hidden)
        return output, hidden

    def initHidden(self, MINI_BATCH):
        result1 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        result2 = Variable(torch.zeros(self.hidden_layer*2, MINI_BATCH, self.hidden_size)) if self.bidir else Variable(torch.zeros(self.hidden_layer, MINI_BATCH, self.hidden_size))
        if use_cuda:
            return (result1.cuda(), result2.cuda())
        else:
            return (result1, result2)

class SDT6Classifier(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, MINI_BATCH, WINDOW_SIZE, BIDIR):
        super(SDT6Classifier, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.hidden_layer = HIDDEN_LAYER
        self.mini_batch  = MINI_BATCH
        self.window_size = WINDOW_SIZE
        self.bidir = BIDIR

        self.attn = nn.Linear(2*HIDDEN_SIZE, WINDOW_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, WINDOW_SIZE)
        self.attn_softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE) if BIDIR else nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attn_relu = nn.ReLU()
        self.dec_ln = nn.LayerNorm(HIDDEN_SIZE)
        self.dec_linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.dec_relu = nn.ReLU()
        self.dec_softmax = nn.Softmax(dim=2)

    def forward(self, attn_hidden, enc_outs):
        attn_weights = self.attn(attn_hidden)
        attn_weights = self.attn_softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_outs)

        output = self.attn_combine(attn_applied)
        output = self.attn_relu(output)
        output = self.dec_ln(output.squeeze(1))
        output = self.dec_linear(output)
        output = self.dec_relu(output)
        output = output.unsqueeze(1)
        output1 = self.dec_softmax(output[:,:,:2])
        output2 = self.dec_softmax(output[:,:,2:4])
        output3 = self.dec_softmax(output[:,:,4:])
    
        return output1, output2, output3, attn_weights