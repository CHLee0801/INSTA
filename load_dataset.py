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
    #with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=config.max_output_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# def dataset_preprocessor(dataset_name, template_name='', type_path = 'train'):
#     dataset_dict = {
#         'quoref' : ['extractive_qa', 'quoref', 'validation', 0],
#         'ropes' : ['extractive_qa', 'ropes', 'validation', 0],
#         'duorc' : ['extractive_qa', 'duorc/ParaphraseRC', 'validation', 0],
#         'adversarial_qa' : ['extractive_qa', 'adversarial_qa/adversarialQA', 'validation', 0],

#         'qqp' : ['paraphrase_identification', 'glue/qqp', 'validation', 1],
#         'mrpc' : ['paraphrase_identification', 'glue/mrpc', 'validation', 1],
#         'paws' : ['paraphrase_identification', 'paws/labeled_final', 'validation', 1],

#         'ag_news' : ['topic_classification', 'ag_news', 'test', 1],
#         'trec' : ['topic_classification', 'trec', 'test', 1],
#         'dbpedia' : ['topic_classification', 'dbpedia_14', 'test', 1],

#         'imdb' : ['sentiment', 'imdb', 'test', 1],
#         'yelp' : ['sentiment', 'yelp_review_full', 'test', 1],
#         'amazon' : ['sentiment', 'amazon_polarity', 'test', 1],
#         'app_reviews' : ['sentiment', 'app_reviews', 'test', 1],
#         'rotten_tomatoes' : ['sentiment', 'rotten_tomatoes', 'test', 1],

#         'wiki_qa' : ['closed_book_qa', 'wiki_qa', 'validation', 0],
#         'hotpot_qa' : ['closed_book_qa', 'hotpot_qa/distractor', 'validation', 0],

#         'multi_news' : ['summarization', 'multi_news', 'validation', 0],
#         'gigaword' : ['summarization', 'gigaword', 'validation', 0],
#         'samsum' : ['summarization', 'samsum', 'validation', 0],
#         'xsum' : ['summarization', 'xsum', 'validation', 0],
#         'cnn_daily_mail' : ['summarization', 'cnn_dailymail/3.0.0', 'validation', 0],

#         'common_gen' : ['struct_to_text', 'common_gen', 'validation', 0],
#         'wiki_bio' : ['struct_to_text', 'wiki_bio', 'val', 0],

#         'cos_e' : ['multiple_choice_qa', 'cos_e/v1.11', 'validation', 1],
#         'dream' : ['multiple_choice_qa', 'dream', 'test', 1],
#         'quail' : ['multiple_choice_qa', 'quail', 'validation', 1],
#         'social_iqa' : ['multiple_choice_qa', 'social_i_qa', 'validation', 1],
#         'quartz' : ['multiple_choice_qa', 'quartz', 'validation', 1],
#         'wiqa' : ['multiple_choice_qa', 'wiqa', 'validation', 1],
#         'cosmos_qa' : ['multiple_choice_qa', 'cosmos_qa', 'validation', 1],
#         'qasc' : ['multiple_choice_qa', 'qasc', 'validation', 1],
#         'quarel' : ['multiple_choice_qa', 'quarel', 'validation', 1],
#         'sciq' : ['multiple_choice_qa', 'sciq', 'validation', 1],
#         'wiki_hop' : ['multiple_choice_qa', 'wiki_hop/original', 'validation', 1],
        
#         'wic' : ['word_sense_disambiguation', 'super_glue/wic', 'validation', 1],

#         'anli_r1' : ['natural_language_inference', 'anli', 'dev_r1', 1],
#         'anli_r2' : ['natural_language_inference', 'anli', 'dev_r2', 1],
#         'anli_r3' : ['natural_language_inference', 'anli', 'dev_r3', 1],
#         'cb' : ['natural_language_inference', 'super_glue/cb', 'validation', 1],
#         'rte' : ['natural_language_inference', 'super_glue/rte', 'validation', 1],

#         'winogrande' : ['coreference_resolution', 'winogrande/winogrande_xl', 'validation', 1],
#         'wsc' : ['coreference_resolution', 'super_glue/wsc.fixed', 'validation', 1],
        
#         'storycloze' : ['sentence_completion', 'story_cloze/2016', 'test', 1],
#         'copa' : ['sentence_completion', 'super_glue/copa', 'validation', 1],
#         'hellaswag' : ['sentence_completion', 'hellaswag', 'validation', 1],
        
#         'boolq' : ['multiple_choice_qa', 'super_glue/boolq', 'validation', 1],
#         'multirc' : ['multiple_choice_qa', 'super_glue/multirc', 'validation', 1],
#         'record' : ['multiple_choice_qa', 'super_glue/record', 'validation', 0],
#         'race' : ['multiple_choice_qa', 'race/all', 'validation', 0],
#         'squad' : ['extractive_qa', 'squad', 'validation', 0],
#         'drop' : ['extractive_qa', 'drop', 'validation', 0],
#         'arc' : ['multiple_choice_qa', 'ai2_arc/ARC-Challenge', 'validation', 1],
#         'piqa' : ['multiple_choice_qa', 'piqa', 'validation', 1],
#         'openbookqa' : ['multiple_choice_qa', 'openbookqa/main', 'validation', 1],
#         'cbt' : ['multiple_choice_qa', 'cbt/CN', 'validation', 1],
#         'art' : ['multiple_choice_qa', 'art', 'validation', 1]
#     }
#     source_list, target_list = [], []
#     dataset_cluster, dataset_name_in_promptsource, test_type_path, classification_or_not = dataset_dict[dataset_name]
    
#     if '/' in dataset_name_in_promptsource:
#         main_name, sub_name = dataset_name_in_promptsource.split('/')
#         if dataset_name == 'storycloze':
#             if type_path == "train":
#                 dataset = load_dataset(main_name, sub_name, data_dir="./data/storycloze/original")["validation"]
#             else:
#                 dataset = load_dataset(main_name, sub_name, data_dir="./data/storycloze/original")[test_type_path]
#         else:
#             if type_path == 'test':
#                 dataset = load_dataset(main_name, sub_name)[test_type_path]
#             else:
#                 dataset = load_dataset(main_name, sub_name)[type_path]
#     else:
#         if type_path == 'test':
#             dataset = load_dataset(dataset_name_in_promptsource)[test_type_path]
#         else:
#             dataset = load_dataset(dataset_name_in_promptsource)[type_path]
            
#     prompt = DatasetTemplates(dataset_name_in_promptsource)
#     prompt_list = list(vars(prompt)['name_to_id_mapping'].keys())

#     idx = 0
#     for example in tqdm(dataset):
#         if template_name == '':
#             target_prompt = random.choice(prompt_list)
#         else:
#             target_prompt = template_name

#         if dataset_name == 'duorc' and example['no_answer'] == True:
#             while target_prompt in ['generate_question_by_answer', 'build_story_around_qa']:
#                 target_prompt = random.choice(prompt_list)
#         if dataset_name == 'wiki_qa' and example['label'] == 0:
#             while target_prompt in ["Jeopardy style", "Topic Prediction - Question and Answer Pair", "Generate Question from Topic", "Topic Prediction - Question Only", "Topic Prediction - Answer Only", "Direct Answer to Question"]:
#                 target_prompt = random.choice(prompt_list)

#         if dataset_name == 'trec':
#             example['label_coarse'] = example['coarse_label']

#             while len(prompt[target_prompt].apply(example)) == 1 or prompt[target_prompt].apply(example)[1] == '':
#                 target_prompt = random.choice(prompt_list)

#         start_time = time.time()
#         flag = 0
#         if dataset_name == 'multi_news':
#             try:
#                 while len(prompt[target_prompt].apply(example)) == 1 or prompt[target_prompt].apply(example)[1] == '':
#                     if time.time() - start_time > 2:
#                         flag = 1 
#                         break
#                     target_prompt = random.choice(prompt_list)
#             except:
#                 continue
#         if flag == 1:
#             continue

#         while len(prompt[target_prompt].apply(example)) == 1 or prompt[target_prompt].apply(example)[1] == '':
#             target_prompt = random.choice(prompt_list)
        
#         if template_name != '' and target_prompt != template_name:
#             continue
        
#         result = prompt[target_prompt].apply(example)
        
#         if dataset_name == 'drop':
#             temp_dict = {}
#             temp_list = result[1].split(', ')
#             max_num = 0
#             if len(temp_list) > 1:
#                 for temp in temp_list:
#                     if temp in temp_dict:
#                         temp_dict[temp] += 1
#                     else:
#                         temp_dict[temp] = 1
#                 for temp in temp_dict:
#                     if temp_dict[temp] > max_num:
#                         max_num = temp_dict[temp] 
#                         result[1] = temp
#         if result[1] == '':
#             continue
        
#         source_list.append(result[0])
#         target_list.append(result[1])

#         idx += 1
#         if idx >= 50000:
#             break
    
#     if len(source_list) != len(target_list):
#         raise Exception(f'{dataset_name} loading error')
    
#     if len(source_list) > 50000:
#         source_list = source_list[:50000]
#         target_list = target_list[:50000]
    
#     return source_list, target_list

def dataset_preprocessor(dataset_name, template_name='', type_path = 'train'):
    source_list, target_list = [], []
    
    target_path = f'data/t0/{dataset_name}/train.csv'
    target_csv_file = pd.read_csv(target_path)
    target_csv_file = target_csv_file.dropna(subset=['input', 'output'])
    # target_csv_file.drop('choices', inplace=True, axis=1)
    # target_csv_file.drop('dataset_name', inplace=True, axis=1)
    # target_csv_file.drop('id', inplace=True, axis=1)
    # train_dataset = load_dataset('csv', data_files=target_path)
    # print(target_csv_file)
    # exit()
    for idx, row in target_csv_file.iterrows():
        source_list.append(row['input'])
        target_list.append(row['output'])
    
    return source_list, target_list

def dataset_preprocessor_super(dataset_name, is_bbh):
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

# def dataset_preprocessor_super(dataset_name, template_name='', type_path = 'train'):
#     source_list, target_list = [], []
#     target_path = f'data/natural-instructions/train/{dataset_name}'
#     train_files = os.listdir(target_path)
#     train_df = pd.DataFrame()
#     for train_file_path in train_files:
#         data_df = pd.read_csv(f'{target_path}/{train_file_path}')
#         train_df = pd.concat([train_df, data_df])
#     target_csv_file = train_df.sample(n=min(50000, train_df.shape[0]), random_state=1004)
#     # target_csv_file = pd.read_csv(target_path)
#     # target_csv_file = target_csv_file.dropna(subset=['input', 'output'])
#     for idx, row in target_csv_file.iterrows():
#         source_list.append(str(row['input']))
#         target_list.append(str(row['output']))
#     return source_list, target_list

class SupervisedDataset(Dataset):
    def __init__(self, dataset_path: Sequence[str], args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        sources, targets = [], []
        
        if args.is_super == True:
            for idx, dataset in enumerate(dataset_path):
                dataset_source, dataset_target = dataset_preprocessor_super(dataset, args.is_bbh)
                sources += dataset_source 
                targets += dataset_target
        else:
            if args.template_name == '':
                for dataset in dataset_path:
                    dataset_source, dataset_target = dataset_preprocessor(dataset)
                    sources += dataset_source 
                    targets += dataset_target
            else:
                for idx, dataset in enumerate(dataset_path):
                    dataset_source, dataset_target = dataset_preprocessor(dataset, args.template_name[idx])
                    sources += dataset_source 
                    targets += dataset_target
            
        data_dict = preprocess(sources, targets, args, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])