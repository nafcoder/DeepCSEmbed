# DeepCSEmbed

### Architecture of InceptionModule (C-linked glycosylation)
![InceptionModule-1](https://github.com/user-attachments/assets/356432f2-7297-4496-9b49-9cedd1fde23f)

### Model architecture of DeepCSEmbed-C (C-linked glycosylation)
![c_linked_framewok-1](https://github.com/user-attachments/assets/dd7addd0-e4f1-42f5-901a-3b2fcaf9bbd2)

### Framework of DeepCSEmbed-S (S-linked glycosylation)
![s_linked_framework-1](https://github.com/user-attachments/assets/8a8e7894-188c-4478-8be9-37a0b362aa5e)

### Data availability
All training and independent datasets are given in [dataset](dataset) folder

### Environments
OS: Pop!_OS 22.04 LTS

Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
pytorch==2.4.1
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
PyQt5==5.15.10
imblearn==0.0
skops==0.9.0
shap==0.45.1
IPython==8.18.1
tqdm==4.66.5
biopython==1.84
transformers==4.44.2
```

### Reproduce results
1. In [c_linked](c_linked) and [s_linked](s_linked), reproducable codes are given. Training scripts are also provided.

### Prediction
#### Prerequisites
1. transformers and Pytorch libaries are needed for extracting the embeddings.

2. For more query, you can visit the following GitHubs:

    [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans)

    [ESM2](https://github.com/facebookresearch/esm)

#### Steps
### C-linked glycosylation
1. Firsly, you need to fillup [dataset.txt](c_linked/Prediction/dataset.txt). Follow the pattern shown below:

```
Protein_id,site_position_1,site_position_2,...,site_position_n
Fasta
```

2. For predicting C-linked glycosylation sites from a protein sequence, you need to run the [extract_features.py](c_linked/Prediction//extract_features.py) to generate features and then run [predict.py](c_linked/Prediction//predict.py) for prediction.

### S-linked glycosylation
1. Firsly, you need to fillup [dataset.txt](s_linked/Prediction/dataset.txt). Follow the pattern shown below:

```
Protein_id,site_position_1,site_position_2,...,site_position_n
Fasta
```

2. For predicting S-linked glycosylation sites from a protein sequence, you need to run the [extract_features.py](s_linked/Prediction//extract_features.py) to generate features and then run [predict.py](s_linked/Prediction//predict.py) for prediction.

### Reproduce previous paper metrics for C-linked glycosylation
In [prev_paper](c_linked/prev_paper), scripts are provided for reproducing the results of the previous papers.
