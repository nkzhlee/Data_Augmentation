'''
https://huggingface.co/models
Most used:

bert-base-uncased
distilbert-base-uncased
roberta-base
google/electra-small-discriminator
YituTech/conv-bert-base
'''

TRAIN_FILE_PATH = ['./data/sst2/train.csv']
TEST_FILE_PATH = ['./data/sst2/validation.csv']
num_labels=2
MAX_SEQ_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 3

hyperparameters = dict(
    model_name="YituTech/conv-bert-base",
    batch_size=32,
    lr=5e-5,
    epochs=5,
    dataset_name="sst2"
    )
# wandb config

config_dictionary = dict(
    #yaml=my_yaml_file,
    params=hyperparameters,
    )


