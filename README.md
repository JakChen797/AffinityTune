# AffinityTune

This is the code associated with the submission "AffinityTune: A Prompt-Tuning Framework for Few-Shot Anomaly Detection on Graphs".

### 1. Dependencies (with python >= 3.8):
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c dglteam/label/cu116 dgl
conda install scikit-learn
pip install pygod
```

### 2. Dataset
For real-world datasets, they can be downloaded from `https://github.com/pygod-team/data` and placed in the "dataset" folder. Also, you can inject anomalies by executing "python inject_ano.py".

### 3. Unsupervised Learning
Run `python pretrain.py ` to perform the first stage of the framework and obtain the trained GNN model.


### 4. Prompt Tuning
Run `python tune.py ` to perform prompt tuning and anomaly detection.
