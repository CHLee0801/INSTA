from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset
import transformers
import torch
from datasets import load_dataset
from promptsource.templates import DatasetTemplates, choice
from tqdm import tqdm 
import random
import time
import pandas as pd
import os

def preprocess(sources, targets, config, tokenizer, padding='max_length'): 
    model_inputs = tokenizer(sources, max_length=config.max_input_length, padding=padding, truncation=True)
    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=config.max_output_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def dataset_preprocessor_p3(dataset_name):
    source_list, target_list = [], []
    
    target_path = f'data/p3/{dataset_name}/train.csv'
    target_csv_file = pd.read_csv(target_path)
    target_csv_file = target_csv_file.dropna(subset=['input', 'output'])

    for idx, row in target_csv_file.iterrows():
        source_list.append(row['input'])
        target_list.append(row['output'])
    
    return source_list, target_list

def dataset_preprocessor_niv2(dataset_name, is_bbh):
    source_list, target_list = [], []
    if is_bbh == True:
        train_df = pd.read_csv(f'data/natural-instructions/train_bbh/{dataset_name}.csv')
    else:
        train_df = pd.read_csv(f'data/natural-instructions/train/{dataset_name}.csv')[:-2]
    target_csv_file = train_df.sample(n=min(5000, train_df.shape[0]), random_state=1004)
    for idx, row in target_csv_file.iterrows():
        source_list.append(str(row['input']))
        target_list.append(str(row['output']))
    return source_list, target_list

class SupervisedDataset(Dataset):
    def __init__(self, dataset_path: Sequence[str], args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        sources, targets = [], []
        
        if args.is_super == True:
            for idx, dataset in enumerate(dataset_path):
                dataset_source, dataset_target = dataset_preprocessor_niv2(dataset, args.is_bbh)
                sources += dataset_source 
                targets += dataset_target
        else:
            for dataset in dataset_path:
                dataset_source, dataset_target = dataset_preprocessor_p3(dataset)
                sources += dataset_source 
                targets += dataset_target
            
        data_dict = preprocess(sources, targets, args, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def label_(name, options, answer):
    label = options.index(answer)
    return label

class InferenceDataset(Dataset):
    def __init__(self, dataset, tokenizer, type_path, input_length, output_length, args, mode='eval'):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path

        self.whole_dataset = type_path

        if mode == 'bbh':
            self.dataset_name = dataset
            self.dataset = pd.read_csv(f'data/bbh_super/{self.dataset_name}/test.csv')
        elif mode == 'eval':
            self.dataset_name = dataset
            self.dataset = pd.read_csv(f'data/natural-instructions/test/{self.dataset_name}.csv')
        else:
            self.dataset = dataset
            self.dataset_name = 'dev'
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        
        self.input_length = 0
        for idx, row in self.dataset.iterrows():
            kkk = len(tokenizer(row['input'])['input_ids'])
            self.input_length = max(self.input_length, kkk)
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_feature_tokenizer(self, input_, target_, options):
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                            padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")

        data_label = self.whole_dataset 
        return source, targets, data_label, options

    def convert_to_features(self, example_batch, index):
        # prompt evaluation
        label = -1

        input_ = example_batch['input']
        target_ = example_batch['output']
        
        try:
            options = example_batch['choices'].split('|||')
            options = [op.strip() for op in options]
        except:
            options = None

        if self.type_path != 'train' and (options != None and self.type_path == 'test'):
            label = options.index(str(target_).strip())

        #print("input:\n", input_)
        #print("target:",target_)
        #print("option and answer is", options, options[label])
        source, targets, data_label, options = self.convert_to_feature_tokenizer(input_, target_, options)
        return source, targets, data_label, options, label

    def __getitem__(self, index):
        indexed_data = self.dataset.iloc[index]    
        #1 only direct (no ul loss case & ul loss validation) & channel base
        source, targets, data_label, options, label = self.convert_to_features(indexed_data, index)
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        if self.type_path != 'train' and (options != None and self.type_path == 'test'):
            option_list = options
        else:
            option_list = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "data_label": data_label, "option_list": option_list, "label": label}