'''
https://huggingface.co/models
Most used:

bert-base-uncased
distilbert-base-uncased
roberta-base
google/electra-small-discriminator
YituTech/conv-bert-base
'''

TRAIN_FILE_PATH = ['./data/TREC/trec_train.csv']
TEST_FILE_PATH = ['./data/TREC/trec_test.csv']
num_labels=6
MAX_SEQ_LENGTH = 64
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 3

hyperparameters = dict(
    model_name="bert-base-cased",
    batch_size=32,
    lr=5e-5,
    epochs=5,
    )
# wandb config

config_dictionary = dict(
    #yaml=my_yaml_file,
    params=hyperparameters,
    )


