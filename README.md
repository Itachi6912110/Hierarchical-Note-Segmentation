# Hierarchical-Note-Segmentation
Realization for note segmentation by using hierarchical objective function

# Loading Data
```
bash/dataset.sh
```

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
