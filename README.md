# Hierarchical-Note-Segmentation
Realization for note segmentation by using hierarchical objective function

# Loading Data
```
bash/dataset.sh
```
This will load the data you need for training and evaluation.
Training data are from TONAS dataset, and testing data are from ISMIR2014 evaluation dataset.
The features used for training is extracted by the tool at https://github.com/leo-so/VocalMelodyExtPatchCNN

# Training
- For training resnet-18
```
bash script/train_sdt6_resnet_top.sh
```
- For training rnn-attn-19
```
bash script/train_sdt6_top.sh
```

# Evaluation
- For evaluating on resnet-18
```
bash script/eval_resnet_fmeasure.sh
```
- For evaluating on rnn-attn-19
```
bash script/eval_sdt6_fmeasure.sh
```

# Visualization
```
python3 plot_P.py <file#>
```
