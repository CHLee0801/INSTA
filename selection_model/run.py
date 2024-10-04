from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
from sentence_transformers import evaluation
import json
import sys
import os
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd 

random.seed(42)

data_path = sys.argv[1]

with open(f'data/{data_path}/train.json', 'r') as file:
    train_data = json.load(file)

with open(f'data/{data_path}/dev.json', 'r') as file:
    dev_data = json.load(file)
    
#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('sentence-transformers/bert-large-nli-stsb-mean-tokens')

train_examples = [InputExample(texts=[d[0], d[1]], label=float(d[2])) for d in train_data]
dev_examples = [InputExample(texts=[d[0], d[1]], label=float(d[2])) for d in dev_data]

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='sts-dev')
#Tune the model
saving_steps = len(train_examples) / 16
saving_steps = int(saving_steps) if len(train_examples) / 16 == len(train_examples)//16 else int(saving_steps) + 1

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, optimizer_params={'lr':1e-6},warmup_steps=saving_steps//10, evaluator=evaluator, checkpoint_path=f'ckpt/{data_path}', checkpoint_save_steps=saving_steps, checkpoint_save_total_limit = 5)

base_path = '/home/ubuntu/ex_disk/MT_Selector/selection_model/ckpt/' + data_path + '/'
dev_examples = [InputExample(texts=[d[0], d[1]], label=float(d[2])) for d in dev_data]

result_list = []

for ckpt in os.listdir(base_path):
    ckpt_path = base_path + ckpt

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_examples,
        name="sts-dev",
    )
    model = SentenceTransformer(ckpt_path).cuda()
    test_evaluator(model, output_path='/home/ubuntu/ex_disk/MT_Selector/selection_model/ckpt/' + data_path + '/' +ckpt)


result_list = []

for ckpt in os.listdir(base_path):
    ckpt_path = base_path + ckpt + '/similarity_evaluation_sts-dev_results.csv'
    a = pd.read_csv(ckpt_path)

    result_list.append([ckpt, a['cosine_pearson'][0]])

result_list.sort(key=lambda x:x[1], reverse=True)

for k in result_list:
    print(k)
    
    