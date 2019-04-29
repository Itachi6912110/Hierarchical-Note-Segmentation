import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import random

use_cuda = torch.cuda.is_available()

def train_sdt4(input_Var, target_Var, encoders, decoders, enc_opts, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onEnc       = encoders[0]
    onDec       = decoders[0]
    onEncOpt    = enc_opts[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 
    
    input_batch = input_Var.size()[0]
    input_time_step = input_Var.size()[1]

    onEncOpt.zero_grad()
    onDecOpt.zero_grad()

    onLoss  = 0

    window_size = 2*k+1
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:
            # Sampling
            if target_Var[:, BATCH_SIZE*step, 0].item() == 1.0 or target_Var[:, BATCH_SIZE*step, 1].item() == 1.0:
                r = random.random()
                if r < 0.5:
                    continue

            onEncHidden = onEnc.initHidden(BATCH_SIZE)
            
            onEncOuts = torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size*2) if onEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size)

            # Onset Encode 
            for ei in range(window_size):
                enc_out, onEncHidden = onEnc(input_Var[:, BATCH_SIZE*step-k+ei:BATCH_SIZE*step-k+ei+BATCH_SIZE, :].contiguous().view(BATCH_SIZE, 1, INPUT_SIZE), onEncHidden)
                onEncOuts[ei] = enc_out.squeeze(1).data

            # To Onset Decoder
            onEncOuts = onEncOuts.transpose(0, 1)

            onDecAttnHidden = torch.cat((onEncHidden[0][2*onEnc.hidden_layer-1], onEncHidden[0][2*onEnc.hidden_layer-2]), 1) if onEnc.bidir else onEncHidden[0][onEnc.hidden_layer-1]
            onEncOuts = Variable(onEncOuts).cuda()

            # 1 step input (cause target only 1 time step)
            onDecOut1, onDecOut2, onDecAttn = onDec(onDecAttnHidden, onEncOuts)

            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut1[i].view(1, OUTPUT_SIZE//2), torch.max(target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(input_batch, 1, OUTPUT_SIZE//2), dim=2)[1].view(input_batch))
                onLoss += onLossFunc(onDecOut2[i].view(1, OUTPUT_SIZE//2), torch.max(target_Var[:,BATCH_SIZE*step+i, 2:].contiguous().view(input_batch, 1, OUTPUT_SIZE//2), dim=2)[1].view(input_batch))
                        
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    onLoss.backward()

    onEncOpt.step()
    onDecOpt.step()

    return onLoss.item() / input_time_step