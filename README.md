# HIGIL

### Code for "HiGIL: Hierarchical Graph Inference Learning for Fact Checking"

## 0 Experiment Environment

The version of Python and essential pypi package requirements are as follows.

    Python==3.8.10
    torch==1.8.1
    numpy==1.19.2
    dgl-cu102==0.7.2
    transformers==4.4.2


## 1 Data Preparation

download datasets from https://competitions.codalab.org/competitions/18814#participate

* ```train.jsonl```: Training Dataset
* ```shared_task_dev.jsonl```: Evaluation Dataset
* ```shared_task_test.jsonl```: Testing Dataset

download constructed graphs from https://drive.google.com/drive/folders/1zJ3ICkiG550oHTTJozew29G7QprlK7q-?usp=sharing

* ```nli_train_graph.pkl```: Training Graph
* ```nli_dev_graph.pkl```: Evaluation Graph
* ```nli_test_graph.pkl```: Testing Graph


download roberta-large model from https://drive.google.com/drive/folders/1BKRrWN8h8B75l-K0mHPfD08q-yEyVJWQ?usp=sharing


## 2 Code Structure and Essential Modules

    ├─data
    │  ├─fever
    │  │  ├─label.json
    │  │  ├─nli_dev_graph.pkl
    │  │  ├─nli_test_graph.pkl
    │  │  └─nli_train_graph.pkl
    │  └─nli
    │     ├─shared_task_dev.jsonl
    │     ├─shared_task_test.jsonl
    │     └─train.jsonl
    └─src
        ├─nli
        │  ├─pretrained_model
        │  │    └─roberta-large
        │  ├─dataloader.py
        │  └─model.py
        ├─utils
        │  ├─__init__.py
        │  ├─common.py
        │  └─fever.py
        └─config.py


* ```src/config.py```   
  
  Model's full hyperparameters and their default values / Storage paths for data and model files

* ```src/nli/model.py```    
  
  Network and model training / evaluation / testing for the fact checking task

* ```src/nli/pretrained_model/roberta-large``` 
  
  Folder for storing roberta-large model
  


## 3 Usage

Run ```python src/nli/model_origin.py --ckpt_name [param_value] --model [param_value] --evaluate [param_value]``` to train the model. The parameters to be set are as follows.

* ```ckpt_name```   
  
  Set the name of folder for storing the model. For example, if we let ckpt_name=='pvs', the model will be stored in ```data/nli/pvs```.

* ```model``` **choices=['train', 'from_ckpt']** 
  
  if model=='train', the model will be trained from scratch, otherwise we will continue training on top of the original model.

* ```evaluate``` **choices=['train', 'dev', 'test']** 
  
   if evaluate=='test', blind test will be performed, otherwise only training and evalution will be performed.


### Example:  Train the model from scratch and perform testing,  the model will be stored in ```data/nli/pvs```.


    python src/nli/model_origin.py --ckpt_name pvs --model train --evaluate test





