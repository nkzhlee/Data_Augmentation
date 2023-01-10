import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, BertForQuestionAnswering, DistilBertForQuestionAnswering
from torch.optim import AdamW
from Constants import *
from DataModules import *
import pdb


def train(args):
    wandb.init(project="Data Augmentation", entity="psu-nlp", config=config_dictionary)
    random.seed(123)
    DEVICE = args.device
    print(DEVICE)
    model_name = hyperparameters['model_name']
    print(model_name)
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    contexts, questions, answers = read_squad(TRAIN_FILE_PATH)
    test_contexts, test_questions, test_answers = read_squad(TEST_FILE_PATH)
    #test_dataset = SequenceDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    trainset_size = len(questions)
    testset_size = len(test_questions)
    shuffle_dataset = True
    validation_split = 0.2
    #indices = list(range(trainset_size))
    split = int(np.floor(validation_split * trainset_size))

    temp = list(zip(contexts, questions, answers))
    

    if shuffle_dataset:
        random.shuffle(temp)

    contexts, questions, answers = zip(*temp)
    contexts, questions, answers = list(contexts), list(questions), list(answers)
    train_contexts, train_questions, train_answers = contexts[split:], questions[split:], answers[split:]
    val_contexts, val_questions, val_answers = contexts[:split], questions[:split], answers[:split]

    ## adding end character information
    train_answers, train_contexts = add_end_idx(train_answers, train_contexts)
    val_answers, val_contexts = add_end_idx(val_answers, val_contexts)
    test_answers, test_contexts = add_end_idx(test_answers, test_contexts)


    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)
    #train_indices, val_indices = indices[split:], indices[:split]

    ## adding start and end position token information
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)
    add_token_positions(test_encodings, test_answers, tokenizer)

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    test_dataset = SquadDataset(test_encodings)



    #train_sampler = SubsetRandomSampler(train_indices)
    #validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'])
    #                                           sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'])
    #                                         sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    training_acc_list, validation_acc_list = [], []

    if model_name == 'bert-base-uncased' :
        model = BertForQuestionAnswering.from_pretrained(model_name)
    elif model_name == 'distilbert-base-uncased' :
        model = DistilBertForQuestionAnswering.from_pretrained(model_name)   
    #model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'])
    model.train()
    # Training Loop

    for epoch in range(hyperparameters['epochs']):
        ## ensuring model is in train set after calculating val accuracy at each epoch
        model.train()
        epoch_loss = 0.0
        train_correct_total = 0
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            #labels = inputs['labels'].to(DEVICE)
            outputs = model(**inputs)
            #logits, loss = outputs.logits, outputs.loss
            loss = outputs[0]
            #loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            epoch_loss += loss.item()

            ''' Calculating train accuracy '''
            start_scores,end_scores = outputs.start_logits, outputs.end_logits

            input_ids = batch['input_ids'].to(DEVICE)
            answer_batch = train_answers[hyperparameters['batch_size']*step:hyperparameters['batch_size']*(step+1)]
            correct_in_batch = 0
            # torch.argmax(start_scores, dim=1)
            start_scores_max = torch.argmax(start_scores,dim=1)
            end_scores_max = torch.argmax(end_scores,dim=1)
            ## getting text output for each example and then comparing with ground truth
            for index,(max_startscore,max_endscore,input_id) in enumerate(zip(start_scores_max,end_scores_max,input_ids)) :
                ans_tokens = input_ids[index][max_startscore: max_endscore + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
                answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

                if answer_batch[index]['text'] == answer_tokens_to_string :
                    correct_in_batch += 1

            train_correct_total += correct_in_batch
            #if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            #    #scheduler.step()
            #    optimizer.step()
            #    model.zero_grad()
            #_, predicted = torch.max(logits.data, 1)
            #correct_reviews_in_batch = (predicted == labels).sum().item()
            #train_correct_total += correct_reviews_in_batch
        print('Epoch {} - Loss {}'.format(epoch + 1, epoch_loss))
        #print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        with torch.no_grad():
            model.eval()
            val_correct_total = 0
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                answer_batch = val_answers[hyperparameters['batch_size']*step:hyperparameters['batch_size']*(step+1)]
                #labels = inputs['labels'].to(DEVICE)
                outputs = model(**inputs)
                #logits, val_loss = outputs.logits, outputs.loss
                ## QA model doesn't have logits
                val_loss = outputs.loss
                #val_loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
                epoch_loss += val_loss.item()
                input_ids = batch['input_ids'].to(DEVICE)
                start_scores,end_scores = outputs.start_logits, outputs.end_logits

                correct_in_batch = 0
                # torch.argmax(start_scores, dim=1)
                start_scores_max = torch.argmax(start_scores,dim=1)
                end_scores_max = torch.argmax(end_scores,dim=1)
                ## getting text output for each example and then comparing with ground truth
                for index,(max_startscore,max_endscore,input_id) in enumerate(zip(start_scores_max,end_scores_max,input_ids)) :
                    ans_tokens = input_ids[index][max_startscore: max_endscore + 1]
                    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
                    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

                    if answer_batch[index]['text'] == answer_tokens_to_string :
                        correct_in_batch += 1

                #_, predicted = torch.max(logits.data, 1)
                #correct_reviews_in_batch = (predicted == labels).sum().item()
                val_correct_total += correct_in_batch
                #break
            #training_acc_list.append(train_correct_total * 100 / len(train_indices))
            #validation_acc_list.append(val_correct_total * 100 / len(val_indices))
            print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
                train_correct_total * 100 / len(train_answers), val_correct_total * 100 / len(val_answers)))
            wandb.log({"Train loss": epoch_loss, "Val loss": val_loss,
                       "Train Acc": train_correct_total * 100 / len(train_answers), "Val Acc": val_correct_total * 100 / len(val_answers)})
    with torch.no_grad():
        test_correct_total = 0
        model.eval()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        for step, batch in enumerate(test_iterator):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            answer_batch = test_answers[step:step+1]
            #labels = inputs['labels'].to(DEVICE)
            outputs = model(**inputs)
                #logits, val_loss = outputs.logits, outputs.loss
                ## QA model doesn't have logits
                #val_loss = outputs.loss
                #val_loss = criterion(logits, labels) / GRADIENT_ACCUMULATION_STEPS
                #epoch_loss += val_loss.item()
            input_ids = batch['input_ids'].to(DEVICE)
            start_scores,end_scores = outputs.start_logits, outputs.end_logits
            start_scores_max = torch.argmax(start_scores)
            end_scores_max = torch.argmax(end_scores)
            ans_tokens = input_ids[0][start_scores_max:end_scores_max+1]
            answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

            if answer_batch[0]['text'] == answer_tokens_to_string : 
                test_correct_total += 1
        #validation_acc_list.append(test_correct_total * 100 / testset_size)
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
