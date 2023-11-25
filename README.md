# SpiderTags

## Introduction
This repo is the source code for paper “Observation is Reality? A Graph Diffusion-Based Approach for Service Tags Recommendation”

The code is oranized as follows: 
```
root
├─SRaSLR-Compare	       compare SpiderTags with SRaSLR to verify the
adaptability of the model   
│  ├─data			       dataset SSN for adaptability experiment
│  ├─Compare.py		       code for SpiderTags to run single-tag recommdation task
│  └─SRaSLR-MM-AA.py           code for SRaSLR
|
└─SpiderTag
    ├─baselines	               methods trained only recommendation task
    ├─Data                     dataset TCN for experiment      
    └─SpiderTags.py	       code for SpiderTags to run multi-tags recommdation task

```

## Usage
### Requirements
The following packages are required:

```
torch==1.10.0
pytorch_lightning==1.5.7
numpy==1.19.5
gensim==3.8.3
nltk==3.6.5
scikit_learn==1.0.2
transformers==4.15.0
torchmetrics==0.6.2
torch_geometric==2.0.4
networkx==2.1.0
dgl==1.1.0
```
### Dataset
The dataset TCN can download from [here](https://www.aliyundrive.com/s/YfhdTs2SYUj).


### Train models
- Clone this project.
- Go into the root of repo and install the required package listed in `requirements.txt` by:
```commandline
pip install -r requirement.txt
```
- Download `bert_model` and put it in `SRaSLR-Compare`.
- Download `myData` in [here](https://www.aliyundrive.com/s/YfhdTs2SYUj) and put it in `SpiderTag`.
- Use `python` command to train and test the model. For example:
```commandline
python TagTag/TagTag.py
```

