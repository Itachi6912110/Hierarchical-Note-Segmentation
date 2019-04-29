#!/bin/bash

# create data folders
mkdir -p data
mkdir -p ans
mkdir -p pitch
mkdir -p output
mkdir -p output/est
mkdir -p output/single
mkdir -p output/total
mkdir -p log
mkdir -p loss


# get training data (preprocessed)
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1FqWzk6qWEp80RWbFV4MCAhmSXX07uUEy" -O data/TONAS_note.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1Fo1yL3S0KnJ3KcOcK7x5T-KTmjDc0jO1" -O data/ISMIR2014_note.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=132b_iGgMtVA9bNmsPGJDo8_k9EfAnkpJ" -O ans/TONAS_SDT6.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1ITHDTxLYLie02z-sD_823phvNcg8-Lc1" -O ans/TONAS_ans.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1_fxgSF1-g5c9G_lG7Cq_3ZQ_sueZTf02" -O ans/ISMIR2014_ans.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=16HwBuiqth_MFTusUw3wrms_yWdu2pjnF" -O pitch/ISMIR2014.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=17igMrG4guOBmb-dKzJr9I-GdxOWa1yG3" -O pitch/TONAS.zip

# unzip all files
unzip data/TONAS_note.zip -d data
unzip data/ISMIR2014_note.zip -d data
unzip ans/TONAS_ans.zip -d ans
unzip ans/TONAS_SDT6.zip -d ans
unzip ans/ISMIR2014_ans.zip -d ans
unzip pitch/ISMIR2014.zip -d pitch
unzip pitch/TONAS.zip -d pitch