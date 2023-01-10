import torch
import csv, json
from torch.utils.data import Dataset
from pathlib import Path
from Constants import *
import pdb

def read_squad(path):
    path = Path(path[0])
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

    return answers, contexts



class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, device):
        # Read JSON file and assign to headlines variable (list of strings)
        self.data_dict = []
        self.device = device
        self.lable_set = set()
        file_data = []
        for file in dataset_file_path:
            with open(file) as csvfile:
                csv_reader = csv.reader(csvfile)
                #file_header = next(csv_reader)
                for row in csv_reader:
                    file_data.append(row)

        for row in file_data:
            data = []
            self.lable_set.add(row[0])
            data.append(row[0])
            data.append(row[1])
            self.data_dict.append(data)
        self.tokenizer = tokenizer
        self.tag2id = self.set2id(self.lable_set)
        print(self.tag2id)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        DEVICE = self.device
        input = {}
        label, line = self.data_dict[index]
        label = self.tag2id[label]
        tokens = self.tokenizer(line, padding="max_length", truncation=True)
        input['labels'] = label
        for k, v in tokens.items():
            input[k] = torch.tensor(v, dtype=torch.long, device=DEVICE)

        return input


    def set2id(self, item_set, pad=None, unk=None):
        item2id = {}
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id
