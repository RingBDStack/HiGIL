import os
import torch
from pathlib import Path


PROJECT_ROOT = Path(os.environ['PROJECT_ROOT'])


###############################
#        Original Data        #
###############################
FEVER_LABEL_PATH = PROJECT_ROOT / 'data' / 'fever' / 'label.json'
FEVER_TRAIN_PATH = PROJECT_ROOT / 'data' / 'fever' / 'train.jsonl'
FEVER_DEV_PATH = PROJECT_ROOT / 'data' / 'fever' / 'shared_task_dev.jsonl'
FEVER_TEST_PATH = PROJECT_ROOT / 'data' / 'fever' / 'shared_task_test.jsonl'

WIKI_DB_PATH = PROJECT_ROOT / 'data' / 'wiki.db'


####################################
#        Document Retrieval        #
####################################
# Retrieved Document Number
DOC_NUM = 10

# Existing Methods
NSMN_DOC_FIELD = 'predicted_docids'
NSMN_DOC_DIR = PROJECT_ROOT / 'data' / 'doc_retr' / 'nsmn'

UKP_DOC_FIELD = 'predicted_pages'
UKP_DOC_DIR = PROJECT_ROOT / 'data' / 'doc_retr' / 'ukp-athene'

# Document Retrieval Result
DOC_FIELD = NSMN_DOC_FIELD
TRAIN_DOC_RETR_PATH = NSMN_DOC_DIR / 'doc_retr_2_train.jsonl'
DEV_DOC_RETR_PATH = NSMN_DOC_DIR / 'doc_retr_2_shared_task_dev.jsonl'
TEST_DOC_RETR_PATH = NSMN_DOC_DIR / 'doc_retr_2_shared_task_test.jsonl'

# Logging
DOC_RETR_LOGGING_DIR = PROJECT_ROOT / 'logging' / 'doc_retr'


####################################
#        Sentence Selection        #
####################################
# Hyperparameter
SENT_SEL_HIDDEN_SIZE = 2048
SENT_SEL_DROPOUT = 0.1

# Experiment
SENT_SEL_TRAIN_EPOCHS = 5
SENT_SEL_TRAIN_BATCH_SIZE = 32
SENT_SEL_EVAL_BATCH_SIZE = 1024

# Negative Sampling
NUM_NEGATIVE = 5

# Sentence Not Found
NO_SENT = '[NO SENT]'
NO_SENT_INDEX = ('[None]', 0)

# Selected Sentence Number
SENT_NUM = 5

# Model Checkpoint
SENT_SEL_MODEL_DIR = PROJECT_ROOT / 'model' / 'sent_sel'

# Sentence Selection Training Data
SENT_SEL_TRAIN_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'sent_sel_train.pkl'
SENT_SEL_DEV_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'sent_sel_dev.pkl'
SENT_SEL_TEST_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'sent_sel_test.pkl'
SENT_SEL_TRAIN_EVAL_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'sent_sel_train_eval.pkl'

# Sentence Selection With Different Scorer Architectures
SENT_SEL_DIR = PROJECT_ROOT / 'data' / 'sent_sel'

# Sentence Selection Result
SENT_FIELD = 'predicted_evidence'
TRAIN_SENT_SEL_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'train_sent_sel.jsonl'
DEV_SENT_SEL_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'dev_sent_sel.jsonl'
TEST_SENT_SEL_PATH = PROJECT_ROOT / 'data' / 'sent_sel' / 'test_sent_sel.jsonl'

# Sentence Selection Evaluation Result
SENT_SEL_RESULT_DIR = PROJECT_ROOT / 'result' / 'sent_sel'

# Logging
SENT_SEL_LOGGING_DIR = PROJECT_ROOT / 'logging' / 'sent_sel'


############################################
#        Natural Language Inference        #
############################################
# Experiment
NLI_TRAIN_EPOCHS = 3    # training epoch
NLI_TRAIN_BATCH_SIZE = 8    # batchsize for training
NLI_INFERENCE_BATCH_SIZE = 32   # batchsize for dev & test
NLI_TRAIN_LR = 1e-5     # learning rate
NLI_TRAIN_LR_BERT = 2e-6  # learning rate
NLI_TRAIN_LR_GRAPH = 2e-3

# Model Checkpoint
NLI_MODEL_DIR = PROJECT_ROOT / 'model' / 'nli2'

# Natural Language Inference Original Training Data
EVID_FIELD = 'evidence'
NLI_TRAIN_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_train.pkl'
NLI_DEV_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_dev.pkl'
NLI_TEST_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_test.pkl'

# Natural Language Inference Data Preprocessed by SRL
SRL_BATCH_SIZE = 64
NLI_TRAIN_SRL_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_train_srl.pkl'
NLI_DEV_SRL_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_dev_srl.pkl'
NLI_TEST_SRL_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_test_srl.pkl'

# Natural Language Inference Data Preprocessed Into Graph
NLI_TRAIN_GRAPH_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_train_graph.pkl'
NLI_DEV_GRAPH_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_dev_graph.pkl'
NLI_TEST_GRAPH_PATH = PROJECT_ROOT / 'data' / 'nli' / 'nli_test_graph.pkl'

# Natural Language Inference With Different Architecture
NLI_DIR = PROJECT_ROOT / 'data' / 'nli'

# Natural Language Inference Result
LABEL_FIELD = 'predicted_label'
TRAIN_NLI_PATH = PROJECT_ROOT / 'data' / 'nli' / 'train_nli.jsonl'
DEV_NLI_PATH = PROJECT_ROOT / 'data' / 'nli' / 'dev_nli.jsonl'
TEST_NLI_PATH = PROJECT_ROOT / 'data' / 'nli' / 'test_nli.jsonl'

# Logging
NLI_LOGGING_DIR = PROJECT_ROOT / 'logging' / 'nli'


########################
#        Device        #
########################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#########################
#        Results        #
#########################
FORMAT_DIR = PROJECT_ROOT / 'result' / 'format'
FEVER_STATISTICS_DIR = PROJECT_ROOT / 'result' / 'statistics' / 'fever'
WIKI_STATISTICS_DIR = PROJECT_ROOT / 'result' / 'statistics' / 'wiki'
