# onoffset_sdt.py
# onset & offset detection using Seq2seq AE
# Input: feat, Output: vector(note classification)

import torch
from torch import nn
from onoffset_modules import *
from train_modules.train_sdt6_resnet import train_resnet_4loss
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torchsummary import summary
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import sys
from argparse import ArgumentParser

#----------------------------
# Parser
#----------------------------
parser = ArgumentParser()
parser.add_argument("-d1", help="data file 1 position", dest="d1file", default="data.npy", type=str)
parser.add_argument("-a1", help="label file 1 position", dest="a1file", default="ans1.npy", type=str)
#parser.add_argument("-a2", help="label file 2 position", dest="a2file", default="ans2.npy", type=str)
parser.add_argument("-em1", help="encoder model 1 destination", dest="em1file", default="model/onset_v4_enc", type=str)
parser.add_argument("-emt1", help="train encoder model 1 destination", dest="emt1file", default="model/onset_v4_tenc", type=str)
parser.add_argument("-dm1", help="decoder model 1 destination", dest="dm1file", default="model/onset_v4_dec", type=str)
parser.add_argument("-dmt1", help="train decoder model 1 destination", dest="dmt1file", default="model/onset_v4_tdec", type=str)
parser.add_argument("-p", help="present file number", dest="present_file", default=0, type=int)
parser.add_argument("-e", help="present epoch", dest="present_epoch", default=0, type=int)
parser.add_argument("-l", help="learning rate", dest="lr", default=0.001, type=float)
parser.add_argument("--window-size", help="input window size", dest="window_size", default=3, type=int)
parser.add_argument("--single-epoch", help="single turn training epoch", dest="single_epoch", default=5, type=int)
parser.add_argument("--batch-size", help="training batch size (frames)", dest="batch_size", default=10, type=int)
parser.add_argument("--feat1", help="feature cascaded", dest="feat_num1", default=1, type=int)
parser.add_argument("--hs1", help="latent space size 1", dest="hidden_size1", default=50, type=int)
parser.add_argument("--hl1", help="LSTM layer depth 1", dest="hidden_layer1", default=3, type=int)
parser.add_argument("--bi1", help="LSTM bidirectional switch1", dest="bidir1", default=0, type=bool)
parser.add_argument("--norm", help="normalization layer type", choices=["bn","ln","nb","nl","nn","bb", "bl", "lb","ll"], dest="norm_layer", default="nn", type=str)
parser.add_argument("--loss-record", help="loss record file position", dest="lfile", default="loss.npy", type=str)

args = parser.parse_args()

#----------------------------
# Parameters
#----------------------------
on_data_file = args.d1file # Z file
#off_data_file = args.d2file # Z file
on_ans_file = args.a1file  # marked onset/offset/pitch matrix file
#off_ans_file = args.a2file  # marked onset/offset/pitch matrix file
on_enc_model_file = args.em1file # e.g. model_file = "model/onset_v3_bi_k3"
on_enc_model_train_file = args.emt1file # e.g. model_file = "model/onset_v3_bi_k3"
#off_enc_model_file = args.em2file # e.g. model_file = "model/onset_v3_bi_k3"
on_dec_model_file = args.dm1file # e.g. model_file = "model/onset_v3_bi_k3"
on_dec_model_train_file = args.dmt1file # e.g. model_file = "model/onset_v3_bi_k3"
#off_dec_model_file = args.dm2file # e.g. model_file = "model/onset_v3_bi_k3"
loss_file = args.lfile
INPUT_SIZE1 = 174*args.feat_num1
#INPUT_SIZE2 = 174*args.feat_num2
OUTPUT_SIZE = 6
on_HIDDEN_SIZE = args.hidden_size1
#off_HIDDEN_SIZE = args.hidden_size2
on_HIDDEN_LAYER = args.hidden_layer1
#off_HIDDEN_LAYER = args.hidden_layer2
on_BIDIR = args.bidir1
#off_BIDIR = args.bidir2
LR = args.lr
EPOCH = args.single_epoch
DATA_BATCH_SIZE = 1
BATCH_SIZE = args.batch_size
WINDOW_SIZE = args.window_size
on_NORM_LAYER = args.norm_layer
PATIENCE = 700

PRESENT_FILE = args.present_file
PRESENT_EPOCH = args.present_epoch

#----------------------------
# Data Collection
#----------------------------
print("Training File ", PRESENT_FILE, "| Turn: ", PRESENT_EPOCH)
try:
    myfile = open(on_data_file, 'r')
except IOError:
    print("Could not open file ", on_data_file)
    exit()

with open(on_data_file, 'r') as fd1:
    with open(on_ans_file, 'r') as fa1:
        on_data_np = np.loadtxt(fd1)
        on_data_np = np.transpose(on_data_np)
        #off_data_np = np.loadtxt(fd2)
        #off_data_np = np.transpose(off_data_np)
        on_ans_np = np.loadtxt(fa1, delimiter=',')
        #off_ans_np = np.loadtxt(fa2, delimiter=',')
        min_row = on_ans_np.shape[0] if (on_ans_np.shape[0] < on_data_np.shape[0]) else on_data_np.shape[0]
        on_data_np = on_data_np[:min_row].reshape((1, -1, int(args.feat_num1//3), 174*3)).transpose((0,2,3,1))
        #off_data_np = off_data_np[:min_row].reshape((1,-1))
        on_ans_np = on_ans_np[:min_row].reshape((1,-1))
        #off_ans_np = off_ans_np[:min_row].reshape((1,-1))
        on_data = torch.from_numpy(on_data_np).type(torch.FloatTensor).cuda()
        #off_data = torch.from_numpy(off_data_np).type(torch.FloatTensor).cuda()
        on_ans = torch.from_numpy(on_ans_np).type(torch.FloatTensor).cuda()
        #off_ans = torch.from_numpy(off_ans_np).type(torch.LongTensor).cuda()

#train data
train_loader = data_utils.DataLoader(
    ConcatDataset(on_data, on_ans), 
    batch_size=BATCH_SIZE,
    shuffle=False)

# load resnet50
resnet18 = models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, OUTPUT_SIZE)
num_fout = resnet18.conv1.out_channels
resnet18.conv1 = nn.Conv2d(int(args.feat_num1//3), num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18.avgpool = nn.AvgPool2d(kernel_size=(17,1), stride=1, padding=0)

#----------------------------
# Model Initialize
#----------------------------
if PRESENT_FILE == 1 and PRESENT_EPOCH == 0:
    print("Re-initialize Deep LSTM Module...")
    on_note_decoder = resnet18

else:
    on_note_decoder = resnet18
    on_note_decoder.load_state_dict(torch.load(on_dec_model_train_file))

on_note_decoder.cuda()

note_decoders = [on_note_decoder]

on_dec_optimizer = torch.optim.Adam(on_note_decoder.parameters(), lr=LR)

dec_optimizers = [on_dec_optimizer]

on_loss_func = torch.nn.BCELoss() # maybe should try crossentropy: CrossEntropyLoss()
#on_loss_func = torch.nn.CrossEntropyLoss() # maybe should try crossentropy: CrossEntropyLoss()
#off_loss_func = torch.nn.CrossEntropyLoss() # maybe should try crossentropy: CrossEntropyLoss()

loss_funcs = [on_loss_func]

#----------------------------
# Train Model
#----------------------------
min_loss = 10000.0
stop_count = 0
loss_list = []

for epoch in range(EPOCH):
    total_loss = 0
    loss_count = 0
    for step, xys in enumerate(train_loader):                 # gives batch data
        b_x1 = xys[0].contiguous() # reshape x to (batch, C, feat_size, time_frame)
        b_y1 = Variable(xys[1].contiguous().view(DATA_BATCH_SIZE, -1, OUTPUT_SIZE)).cuda() # batch y
        loss = train_resnet_4loss(b_x1, b_y1, note_decoders, dec_optimizers, loss_funcs,
            INPUT_SIZE1, OUTPUT_SIZE, BATCH_SIZE, k=WINDOW_SIZE)

        total_loss += loss
        loss_count += 1

        avg_loss = total_loss / loss_count

        if(step%10 == 0):
            print('Epoch: ', epoch, '| Step: ', step, '| Loss: %.4f' % loss)

        if avg_loss < min_loss:
            print("Renewing best model ...")
            min_loss = avg_loss
            #torch.save(note_encoders[0].state_dict(), on_enc_model_file)
            torch.save(note_decoders[0].state_dict(), on_dec_model_file)
            #torch.save(note_encoders[1].state_dict(), off_enc_model_file)
            #torch.save(note_decoders[1].state_dict(), off_dec_model_file)
            stop_count = 0
        else:
            stop_count += 1

        if stop_count > PATIENCE:
            print("Early stopping...")
            exit()
    
    loss_list.append(avg_loss)

#torch.save(note_encoders[0].state_dict(), on_enc_model_train_file)
torch.save(note_decoders[0].state_dict(), on_dec_model_train_file)

if PRESENT_EPOCH == 0 and PRESENT_FILE == 1:
    print("Re-initialize Loss Record File ...")
    with open(loss_file, 'w') as flo:
        for i in range(len(loss_list)):
            flo.write("{:.5f}\n".format(loss_list[i]))
else:
    print("Writing Loss to Loss Record File ...")
    with open(loss_file, 'a') as flo:
        for i in range(len(loss_list)):
            flo.write("{:.5f}\n".format(loss_list[i]))
