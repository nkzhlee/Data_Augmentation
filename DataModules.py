import torch
import csv
#import pandas as pd
from torch.utils.data import Dataset

from Constants import *

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