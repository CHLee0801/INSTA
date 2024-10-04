import os
import argparse
from argparse import ArgumentParser
import json
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from tqdm import tqdm
import csv
from datasets import Dataset, load_from_disk, load_metric
import torch
import evaluate
import nltk
import numpy as np
import multiprocessing
import wandb
from load_dataset import SupervisedDataset
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments
from third_party.trainers import Seq2SeqTrainer
from third_party.trainers import TaskDataCollatorForSeq2Seq
from third_party.trainers import PostProcessor
from setproctitle import *

setproctitle('Changho MT Experiment')

local_rank = int(os.getenv('LOCAL_RANK', '0'))
os.environ["WANDB_DISABLED"] = "true"

def check_args(config, checkpoint_path_through_command, dataset_path_through_command, target_cluster, target_dataset, is_bbh, is_super):

    """Check the configurations"""
    # REQUIRED configs
    if 'mode' not in config:
        raise Exception('Please provide the mode of the run. Choose between `train` & `eval`.')
    if 'model_id' not in config:
        raise Exception('Please provide the model_id provide in huggingface models')
    #if 'dataset_path' not in config and dataset_path_through_command == None:
    #    raise Exception('Please provide the dataset path that contains train.json & eval.json')
    if 'epochs' not in config:
        raise Exception('Please provide the epoch of the training data')
    if 'output_dir' not in config:
        raise Exception('Please provide the output directory to save the log files & model checkpoint')
    
    # DEFAULT values for other configs
    if 'per_device_train_batch_size' not in config:
        config.per_device_train_batch_size = 1 # Batch size to use for training.
    if 'per_device_eval_batch_size' not in config:
        config.per_device_eval_batch_size = 1 # Batch size to use for testing.
    if 'max_input_length' not in config:
        config.max_input_length = 768 # Maximum length to use for generation
    if 'max_output_length' not in config:
        config.max_output_length = 256 # Maximum length to use for generation
    if 'generation_num_beams' not in config:
        config.generation_num_beams = 1 # Number of beams to use for generation
    if 'lr' not in config:
        config.lr = 1e-5 # Learning rate to use for training.
    if 'seed' not in config:
        config.seed = 42 # Random seed for all things random
    if 'deepspeed' not in config:
        config.deepspeed = "deepspeed_configs/z3_bf16.json" # Directory to the deepspeed configuration. Details in https://www.deepspeed.ai/tutorials/zero/
    if 'gradient_checkpointing' not in config:
        config.gradient_checkpointing = True # Whether to use gradient checkpointing. 
    if 'bf16' not in config:
        config.bf16 = True if torch.cuda.get_device_capability()[0] == 8 else False # Whether to use bf16.
    if 'num_workers' not in config:
        config.num_workers = multiprocessing.cpu_count()
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'dataset_length' not in config:
        config.dataset_length = 100000
    if 'n_gpu' not in config:
        config.n_gpu = 16
    if 'checkpoint_path' not in config:
        config.checkpoint_path = ""
    if 'template_name' not in config:
        config.template_name = ""

    config.target_cluster = target_cluster
    config.target_dataset=  target_dataset
    
    if checkpoint_path_through_command != None:
        config.checkpoint_path = checkpoint_path_through_command
    if dataset_path_through_command != None:
        config.dataset_path = dataset_path_through_command
    
    # etc.
    if 'repository_id' not in config:
        config.repository_id = None # Hugging Face Repository id for uploading models
    if 'hf_token' not in config:
        config.hf_token = HfFolder.get_token() # Token to use for uploading models to Hugging Face Hub.
    if 'wandb' not in config:
        config.wandb = False
    if 'wandb_entity' not in config:
        config.wandb_entity = 'changholee' # Default wandb entity to log experiments to. Change with your wandb entity
    if 'wandb_project' not in config:
        config.wandb_project = 'mt5_flm' # Change depending on your project name
    if 'wandb_run_name' not in config and config.wandb == True:
        if config.checkpoint_path != "":
            config.wandb_run_name = config.checkpoint_path + '-' + config.dataset_path # Provide name to the run
        else:
            config.wandb_run_name = config.dataset_path

    if 'is_bbh' not in config:
        config.is_bbh = is_bbh

    if 'is_super' not in config:
        config.is_super = is_super
    return config

nltk.download("punkt", quiet=True)

# Metric
#metric = evaluate.load("rouge")
metric = evaluate.load("accuracy")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def training_run(args):
    # Set Random Seed :)
    set_seed(args.seed)

    # Load model & tokenizer from huggingface
    if args.checkpoint_path != "":
        if local_rank == 0:
            print(args.checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            #args.model_id,
            args.checkpoint_path,
            use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
            low_cpu_mem_usage=True
        )    
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
            low_cpu_mem_usage=True
        )   

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    train_dataset, eval_datasets = [],[]

    # Load train & eval datasets
    if args.mode == 'train':
        train_dataset = SupervisedDataset(args.dataset_lists, args, tokenizer)
    else:
        if args.is_bbh == True:
            if args.target_dataset == '':
                eval_path_lists = []
                base_path = 'data/bigbench/'
                task_list = os.listdir(base_path) + ['logical_deduction/' + i for i in os.listdir(f'{base_path}logical_deduction')] + ['conceptual_combinations/' + i for i in os.listdir(f'{base_path}conceptual_combinations')]
                eval_temp_list = [f'{base_path}{i}/test' for i in task_list]
                eval_path_lists = []
                for path in eval_temp_list:
                    if '.py' in path or 'logical_deduction/test' in path or 'conceptual_combinations/test' in path or 'README' in path or '.json' in path:
                        continue
                    eval_path_lists.append(path)
                    eval_datasets.append(load_from_disk(path))
            elif args.target_dataset == 'conceptual_combinations' or args.target_dataset == 'logical_deduction':
                eval_path_lists = []
                base_path = f'data/bigbench/{args.target_dataset}/'
                for k in os.listdir(base_path):
                    if '.json' in k or 'README' in k:
                        continue
                    path = base_path+k+'/test'
                    eval_path_lists.append(path)
                    eval_datasets.append(load_from_disk(path))
            else:
                eval_path_lists = []
                path = f'data/bigbench/{args.target_dataset}/test'
                eval_path_lists.append(path)
                eval_datasets.append(load_from_disk(path))
        else:
            eval_path_lists = []
            folder_path = 'data/eval'
            if args.target_cluster != '':
                cluster_list = [f'data/eval/{args.target_cluster}']
            else:
                cluster_list = ['data/eval/'+f for f in os.listdir(folder_path)]
                
            dataset_path = [os.listdir(d) for d in cluster_list]
            dataset_list = []
            for i in range(len(cluster_list)):
                for j in range(len(dataset_path[i])):
                    if args.target_dataset != "":
                        if args.target_dataset == dataset_path[i][j]:
                            dataset_list.append(cluster_list[i]+'/'+dataset_path[i][j])
                    else:
                        dataset_list.append(cluster_list[i]+'/'+dataset_path[i][j])
            eval_path_lists = []

            for d in dataset_list:
                dd_list = os.listdir(d)
                for ddd in dd_list:
                    if 'csv' in ddd:
                        continue
                    eval_path_lists.append(d+'/'+ddd)

            eval_path_lists.sort()

            for path in eval_path_lists:
                eval_datasets.append(load_from_disk(path))
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = TaskDataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    def get_accuracy(preds, labels):
        total_cnt = 0
        correct = 0
        for i in range(len(preds)):
            total_cnt+=1
            if preds[i] == labels[i]:
                correct+=1
        return {'accuracy': correct / total_cnt}

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        post_processor = PostProcessor(tokenizer, ignore_pad_token_for_loss=True)
        decoded_preds, decoded_labels = post_processor.process(preds, labels)
        result = get_accuracy(preds=decoded_preds, labels=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        if local_rank == 0:
            print(result)
            log_path = 'out/log/poc.csv'
            f = open(log_path, 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            if args.checkpoint_path == '':
                wr.writerow([args.model_id, args.current_dataset, result['accuracy']])
            else:
                wr.writerow([args.checkpoint_path, args.current_dataset, result['accuracy']])
            f.close()  
        return result

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]

    ckpt_saving_steps = len(train_dataset) // (args.gradient_accumulation_steps * args.per_device_train_batch_size * args.n_gpu)
    ckpt_saving_steps *= 5 
    ckpt_saving_steps -= ckpt_saving_steps%10
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_output_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        #sharded_ddp=True,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="no",
        # save_steps=ckpt_saving_steps,
        # save_strategy="steps",
        # save_total_limit=3,
        save_strategy="epoch",
        save_total_limit=3,
        # load_best_model_at_end=True,
        # push to hub parameters
        # report_to="wandb",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adafactor=True
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if args.mode=='train':
        print('Starting Training!')
        trainer.train()

        # Save our tokenizer and create model card
        tokenizer.save_pretrained(output_dir)
        trainer.create_model_card()

        # Push the results to the hub
        if args.repository_id:
            trainer.push_to_hub()
    elif args.mode=='eval':
        if local_rank == 0:
            print('Starting Evaluation!')
        
        for idx, eval_dataset in enumerate(eval_datasets):
            args.current_dataset = eval_path_lists[idx]
            if local_rank == 0:
                print(args.current_dataset)
            trainer.evaluate(eval_dataset = eval_dataset, metric_key_prefix="eval", config=args)
    else:
        raise Exception('Currently only supporting train & eval.')

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--target_cluster', default='', type=str)
    parser.add_argument('--target_dataset', default='', type=str)
    parser.add_argument('--is_bbh', action='store_true')
    parser.add_argument('--is_super', action='store_true')
    arg_, _ = parser.parse_known_args()

    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = check_args(argparse.Namespace(**config), arg_.checkpoint_path, arg_.dataset_path, arg_.target_cluster, arg_.target_dataset, arg_.is_bbh, arg_.is_super)

    if config.wandb:
        wandb.init(entity=config.wandb_entity, project=config.wandb_project, name=config.wandb_run_name)
    training_run(config)

if __name__ == "__main__":
    main()