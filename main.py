import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer
from torch.optim import AdamW
from Constants import *
from DataModules import SequenceDataset


def train(args):
    wandb.init(project="Data Augmentation", entity="psu-nlp", config=config_dictionary)
    random.seed(123)
    DEVICE = args.device
    print(DEVICE)
    model_name = hyperparameters['model_name']
    print(model_name)
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load Train dataset and split it into Train and Validation dataset
    train_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer, DEVICE)
    test_dataset = SequenceDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    trainset_size = len(train_dataset)
    testset_size = len(test_dataset)
    shuffle_dataset = True
    validation_split = 0.2
    indices = list(range(trainset_size))
    split = int(np.floor(validation_split * trainset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'],
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'],
                                             sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)


    training_acc_list, validation_acc_list = [], []

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'])
    model.train()
    # Training Loop

    for epoch in range(hyperparameters['epochs']):
        epoch_loss = 0.0
        train_correct_total = 0
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = inputs['labels'].to(DEVICE)
            outputs = model(**inputs)
            logits, loss = outputs.logits, outputs.loss
            #loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            epoch_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                #scheduler.step()
                optimizer.step()
                model.zero_grad()
            _, predicted = torch.max(logits.data, 1)
            correct_reviews_in_batch = (predicted == labels).sum().item()
            train_correct_total += correct_reviews_in_batch
        print('Epoch {} - Loss {}'.format(epoch + 1, epoch_loss))
        #print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        with torch.no_grad():
            model.eval()
            val_correct_total = 0
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                labels = inputs['labels'].to(DEVICE)
                outputs = model(**inputs)
                logits, val_loss = outputs.logits, outputs.loss
                #val_loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
                epoch_loss += val_loss.item()
                _, predicted = torch.max(logits.data, 1)
                correct_reviews_in_batch = (predicted == labels).sum().item()
                val_correct_total += correct_reviews_in_batch
                #break
            training_acc_list.append(train_correct_total * 100 / len(train_indices))
            validation_acc_list.append(val_correct_total * 100 / len(val_indices))
            print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
                train_correct_total * 100 / len(train_indices), val_correct_total * 100 / len(val_indices)))
            wandb.log({"Train loss": epoch_loss, "Val loss": val_loss,
                       "Train Acc": train_correct_total * 100 / len(train_indices), "Val Acc": val_correct_total * 100 / len(val_indices)})
    with torch.no_grad():
        test_correct_total = 0
        model.eval()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        for step, batch in enumerate(test_iterator):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = inputs['labels'].to(DEVICE)
            outputs = model(**inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits.data, 1)
            correct_reviews_in_batch = (predicted == labels).sum().item()
            test_correct_total += correct_reviews_in_batch
            # break
        validation_acc_list.append(test_correct_total * 100 / testset_size)
        wandb.log({"Test acc": test_correct_total * 100 / testset_size})
        print('Test: \n')
        print('Test Accurracy {:.4f}'.format(test_correct_total * 100 / testset_size))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='debug_cpt',
                        help='ckp_name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
