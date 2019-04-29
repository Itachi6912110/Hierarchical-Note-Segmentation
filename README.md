# Hierarchical-Note-Segmentation
Realization for note segmentation by using hierarchical objective function with utils of Resnet-18 or attention-RNN

# Requirements
Before starting all the tasks bellow, make sure the environment is well set:   
- python 3.5
- pytorch 0.4.1
- torchvision 0.2.1
- mir-eval 0.4

# Demo
For full and real-world demo, please refer to https://github.com/Itachi6912110/WAV2MIDI

# Loading Data
```
bash/dataset.sh
```
This will load the data you need for training and evaluation.   
Training data are from TONAS dataset, and testing data are from ISMIR2014 evaluation dataset.   
The features used for training is extracted by the tool at https://github.com/leo-so/VocalMelodyExtPatchCNN   
Note that the features for training may be too large, and may lead to download abort.

# Training
For training, you can change hyper-parameters in the scripts **train_sdt6_resnet_top.sh** or **train_sdt6_top.sh**
- For training resnet-18
```
bash script/train_sdt6_resnet_top.sh
```
- For training rnn-attn-19
```
bash script/train_sdt6_top.sh
```

# Evaluation
For evaluation, you can change hyper-parameters in the scripts **eval_resnet_fmeasure.sh** or **eval_sdt6_fmeasure.sh**
- For evaluating on resnet-18
```
bash script/eval_resnet_fmeasure.sh
```
- For evaluating on rnn-attn-19
```
bash script/eval_sdt6_fmeasure.sh
```

# Visualization
You can visualize your results by the following command
```
python3 plot_P.py <file#>
```
