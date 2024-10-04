from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import json
import os
import pandas as pd

random.seed(42)

with open('data/dev.json', 'r') as file:
    dev_data = json.load(file)
    
#Define the model. Either from scratch of by loading a pre-trained model

base_path = '/home/changho.lee/MT_Selector/selector_model/ckpt/updated/'
dev_examples = [InputExample(texts=[d[0], d[1]], label=float(d[2])) for d in dev_data]

result_list = []

for ckpt in os.listdir(base_path):
    ckpt_path = base_path + ckpt + '/similarity_evaluation_sts-dev_results.csv'
    a = pd.read_csv(ckpt_path)

    result_list.append([ckpt, a['cosine_pearson'][0]])
"""    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_examples,
        name="sts-dev",
    )
    model = SentenceTransformer(ckpt_path).cuda()
    test_evaluator(model, output_path='/home/changho.lee/MT_Selector/selector_model/ckpt/updated/'+ckpt)
"""

result_list.sort(key=lambda x:x[1], reverse=True)

for k in result_list:
    print(k)