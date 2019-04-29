import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import random

use_cuda = torch.cuda.is_available()

def train_resnet_4loss(input_t, target_Var, decoders, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 
    
    input_batch = input_t.size()[0]
    input_time_step = input_t.size()[3]

    onDecOpt.zero_grad()

    onLoss  = 0

    window_size = 2*k+1

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ input_t[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0))

            onDecOut6 = onDec(input_Var)
            onDecOut1 = nn_softmax(onDecOut6[:, :2])
            onDecOut2 = nn_softmax(onDecOut6[:, 2:4])
            onDecOut3 = nn_softmax(onDecOut6[:, 4:])

            temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
            onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)
            #print(onDecOut4.shape)

            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut1[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut2[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 2:4].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut3[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 4:].contiguous().view(1, 2))
                target_T = torch.max(target_Var[:,BATCH_SIZE*step+i, 3], target_Var[:,BATCH_SIZE*step+i, 5])
                onLoss += onLossFunc(onDecOut4[i].view(1, 3), torch.cat((target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2), target_T.contiguous().view(1, 1)), 1))
                        
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    onLoss.backward()

    onDecOpt.step()

    return onLoss.item() / input_time_step

def train_resnet_2loss(input_t, target_Var, decoders, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 
    
    input_batch = input_t.size()[0]
    input_time_step = input_t.size()[3]

    onDecOpt.zero_grad()

    onLoss  = 0

    window_size = 2*k+1

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ input_t[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0)).cuda()

            onDecOut4 = onDec(input_Var)
            onDecOut1 = nn_softmax(onDecOut4[:, :2]) # onset out
            onDecOut2 = nn_softmax(onDecOut4[:, 2:]) # offset out

            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut1[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 2:4].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut2[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 4:].contiguous().view(1, 2))
                        
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    onLoss.backward()

    onDecOpt.step()

    return onLoss.item() / input_time_step

def train_resnet_sdtloss(input_t, target_Var, decoders, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 
    
    input_batch = input_t.size()[0]
    input_time_step = input_t.size()[3]

    onDecOpt.zero_grad()

    onLoss  = 0

    window_size = 2*k+1

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ input_t[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0)).cuda()

            onDecOut3 = onDec(input_Var)
            onDecOut3 = nn_softmax(onDecOut3) # S, D, T

            for i in range(BATCH_SIZE):
                target_T = torch.max(target_Var[:,BATCH_SIZE*step+i, 3], target_Var[:,BATCH_SIZE*step+i, 5])
                onLoss += onLossFunc(onDecOut3[i].view(1, 3), torch.cat((target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2), target_T.contiguous().view(1, 1)), 1))
                        
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    onLoss.backward()

    onDecOpt.step()

    return onLoss.item() / input_time_step

def train_resnet_2model(input_t, target_Var, decoders, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0]

    #offDec       = decoders[1]
    #offDecOpt    = dec_opts[1]
    #offLossFunc  = loss_funcs[1]
    
    input_batch = input_t.size()[0]
    input_time_step = input_t.size()[3]

    onDecOpt.zero_grad()
    #offDecOpt.zero_grad()

    onLoss  = 0
    #offLoss  = 0

    window_size = 2*k+1

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ input_t[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0)).cuda()

            onDecOut_on = onDec(input_Var)
            onDecOut_on = nn_softmax(onDecOut_on[:, :2]) # onset out

            #onDecOut_off = offDec(input_Var)
            #onDecOut_off = nn_softmax(onDecOut_off[:, :2]) # offset out

            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut_on[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 2:4].contiguous().view(1, 2))
                #offLoss += offLossFunc(onDecOut_off[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 4:].contiguous().view(1, 2))
        
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    onLoss.backward()
    #offLoss.backward()

    onDecOpt.step()
    #offDecOpt.step()

    return onLoss.item() / input_time_step

def train_resnet_3loss(input_t, target_Var, decoders, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 
    
    input_batch = input_t.size()[0]
    input_time_step = input_t.size()[3]

    onDecOpt.zero_grad()

    onLoss  = 0

    window_size = 2*k+1

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ input_t[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0))

            onDecOut6 = onDec(input_Var)
            onDecOut1 = nn_softmax(onDecOut6[:, :2])
            onDecOut2 = nn_softmax(onDecOut6[:, 2:4])
            onDecOut3 = nn_softmax(onDecOut6[:, 4:])

            temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
            onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)
            #print(onDecOut4.shape)

            for i in range(BATCH_SIZE):
                #onLoss += onLossFunc(onDecOut1[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut2[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 2:4].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut3[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 4:].contiguous().view(1, 2))
                target_T = torch.max(target_Var[:,BATCH_SIZE*step+i, 3], target_Var[:,BATCH_SIZE*step+i, 5])
                onLoss += onLossFunc(onDecOut4[i].view(1, 3), torch.cat((target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2), target_T.contiguous().view(1, 1)), 1))
                        
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    onLoss.backward()

    onDecOpt.step()

    return onLoss.item() / input_time_step