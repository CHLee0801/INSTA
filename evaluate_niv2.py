from Datasets_end2end import Pretrain
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import csv
import os 
import re
import string
from rouge import Rouge
from tqdm import tqdm
from collections import Counter
import torch
import numpy as np
from collections import defaultdict
from utils import generation_metric, ids_to_clean_text
import argparse
from argparse import ArgumentParser

def evaluate(args, model, tokenizer, test_dataset):

    model.eval()
    device = "cuda"
    model.to('cuda')

    dataset = Pretrain(dataset=test_dataset, tokenizer=tokenizer, type_path='test', input_length=args.max_input_length, 
                                output_length=args.max_output_length, args=args)


    print('Length of validation data: ',len(dataset))
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,num_workers=4, pin_memory=True)

    total_cnt = 0

    if args.output_log != None:
        f = open(args.output_log, 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
    
    predictions = []
    refs = []
    for batch in tqdm(iter(loader)):
        #print(batch)
        # output_label = batch["label"].tolist()
        with torch.no_grad():
            batch["source_ids"]=batch["source_ids"].to(device)
            batch["source_mask"]=batch["source_mask"].to(device)
            batch["target_mask"]=batch["target_mask"].to(device)
            batch["target_ids"]=batch["target_ids"].to(device)

            # t0 evaluation method - select option with higer probability as answer
            
            outs = model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=args.max_output_length,
                #num_beams=2,
                early_stopping=True,
            )
            dec = ids_to_clean_text(tokenizer, outs)
            targets = ids_to_clean_text(tokenizer, batch['target_ids']) 
            predictions += dec
            refs += targets
            #predictions.append(dec[0])
            #refs.append(targets[0])
            # print(ids_to_clean_text(tokenizer, batch['source_ids'])[0]) 
            # print("TARGET", targets)
            # print("PREDICT", dec)
            # input("keep going?")
        total_cnt+=len(batch['source_ids'])

    final_score = generation_metric(predictions, refs, True)

    if args.checkpoint_path == None:
        first_config = args.model_id
    else:
        first_config = args.checkpoint_path
    if args.output_log != None:
        wr.writerow([first_config, test_dataset, final_score])
    if args.output_log != None:    
        f.close()  

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_id', default='allenai/tk-instruct-3b-def-pos', type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--target_cluster', default='', type=str)
    parser.add_argument('--target_dataset', default='', type=str)
    parser.add_argument('--max_input_length', default=768, type=int)
    parser.add_argument('--max_output_length', default=256, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--output_log', default='out/log/generate.csv', type=str)
    arg_, _ = parser.parse_known_args()

    target_number = '-1'
    if arg_.checkpoint_path != None:
        target_number = arg_.checkpoint_path.split('/')[1].split('_')[-2]
    
    if arg_.checkpoint_path != None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            arg_.checkpoint_path,
            use_cache=False,
            low_cpu_mem_usage=True
        )    
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            arg_.model_id,
            use_cache=False,
            low_cpu_mem_usage=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(arg_.model_id)
    if arg_.target_cluster == '':
        evaluate(arg_, model, tokenizer, arg_.target_dataset)
    else:
        datasets_in_cluster = os.listdir(f'data/natural-instructions/test/{arg_.target_cluster}')
        target_datasets_list = [f'{arg_.target_cluster}/{i}' for i in datasets_in_cluster]
        for target in target_datasets_list:
            if '.csv' in target:
                target = target[:-4]
            if target_number not in target:
                continue
            # if 'task219' not in target and 'task569' not in target and 'task121' not in target and 'task619' not in target and 'task1155' not in target:
            #     continue
            evaluate(arg_, model, tokenizer, target)
            
if __name__ == "__main__":
    main()