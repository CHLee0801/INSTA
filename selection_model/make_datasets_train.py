import json
import itertools
import random
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-large-nli-stsb-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-large-nli-stsb-mean-tokens').cuda()

dataset_dict = {
    'quoref' : ['extractive_qa', 'quoref', 'validation', 0],
    'ropes' : ['extractive_qa', 'ropes', 'validation', 0],
    'duorc' : ['extractive_qa', 'duorc/ParaphraseRC', 'validation', 0],
    'adversarial_qa' : ['extractive_qa', 'adversarial_qa/adversarialQA', 'validation', 0],

    'qqp' : ['paraphrase_identification', 'glue/qqp', 'validation', 1],
    'mrpc' : ['paraphrase_identification', 'glue/mrpc', 'validation', 1],
    'paws' : ['paraphrase_identification', 'paws/labeled_final', 'validation', 1],

    'ag_news' : ['topic_classification', 'ag_news', 'test', 1],
    'trec' : ['topic_classification', 'trec', 'test', 1],
    'dbpedia' : ['topic_classification', 'dbpedia_14', 'test', 1],

    'imdb' : ['sentiment', 'imdb', 'test', 1],
    'yelp' : ['sentiment', 'yelp_review_full', 'test', 1],
    'amazon' : ['sentiment', 'amazon_polarity', 'test', 1],
    'app_reviews' : ['sentiment', 'app_reviews', 'test', 1],
    'rotten_tomatoes' : ['sentiment', 'rotten_tomatoes', 'test', 1],

    'wiki_qa' : ['closed_book_qa', 'wiki_qa', 'validation', 0],
    'hotpot_qa' : ['closed_book_qa', 'hotpot_qa/distractor', 'validation', 0],
    'triviaqa' : ['closed_book_qa', 'trivia_qa/unfiltered', 'validation', 0],
    'webquestions' : ['closed_book_qa', 'web_questions', 'test', 0],

    'multi_news' : ['summarization', 'multi_news', 'validation', 0],
    'gigaword' : ['summarization', 'gigaword', 'validation', 0],
    'samsum' : ['summarization', 'samsum', 'validation', 0],
    'xsum' : ['summarization', 'xsum', 'validation', 0],
    'cnn_daily_mail' : ['summarization', 'cnn_dailymail/3.0.0', 'validation', 0],

    'common_gen' : ['struct_to_text', 'common_gen', 'validation', 0],
    'wiki_bio' : ['struct_to_text', 'wiki_bio', 'val', 0],

    'cos_e' : ['multiple_choice_qa', 'cos_e/v1.11', 'validation', 1],
    'dream' : ['multiple_choice_qa', 'dream', 'test', 1],
    'quail' : ['multiple_choice_qa', 'quail', 'validation', 1],
    'social_iqa' : ['multiple_choice_qa', 'social_i_qa', 'validation', 1],
    'quartz' : ['multiple_choice_qa', 'quartz', 'validation', 1],
    'wiqa' : ['multiple_choice_qa', 'wiqa', 'validation', 1],
    'cosmos_qa' : ['multiple_choice_qa', 'cosmos_qa', 'validation', 1],
    'qasc' : ['multiple_choice_qa', 'qasc', 'validation', 1],
    'quarel' : ['multiple_choice_qa', 'quarel', 'validation', 1],
    'sciq' : ['multiple_choice_qa', 'sciq', 'validation', 1],
    'wiki_hop' : ['multiple_choice_qa', 'wiki_hop/original', 'validation', 1],
    
    'wic' : ['word_sense_disambiguation', 'super_glue/wic', 'validation', 1],

    'anli_r1' : ['natural_language_inference', 'anli', 'dev_r1', 1],
    'anli_r2' : ['natural_language_inference', 'anli', 'dev_r2', 1],
    'anli_r3' : ['natural_language_inference', 'anli', 'dev_r3', 1],
    'cb' : ['natural_language_inference', 'super_glue/cb', 'validation', 1],
    'rte' : ['natural_language_inference', 'super_glue/rte', 'validation', 1],

    'winogrande' : ['coreference_resolution', 'winogrande/winogrande_xl', 'validation', 1],
    'wsc' : ['coreference_resolution', 'super_glue/wsc.fixed', 'validation', 1],
    
    'storycloze' : ['sentence_completion', 'story_cloze/2016', 'test', 1],
    'copa' : ['sentence_completion', 'super_glue/copa', 'validation', 1],
    'hellaswag' : ['sentence_completion', 'hellaswag', 'validation', 1]
}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open('data/train_instances.json', 'r') as file:
    data = json.load(file)

dataset_tmp_dict = {}
task_tmp_dict = {}
device = "cuda:0"
for i in tqdm(data):
    if i['dataset'] not in dataset_tmp_dict:
        dataset_tmp_dict[i['dataset']] = []

    target_text = i['template']
    encoded_input = tokenizer(target_text, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    d = {}
    d['template'] = target_text
    d['embedding'] = embedding

    dataset_tmp_dict[i['dataset']].append(d)
    
    if dataset_dict[i['dataset']][0] not in task_tmp_dict:
        task_tmp_dict[dataset_dict[i['dataset']][0]] = []
    task_tmp_dict[dataset_dict[i['dataset']][0]].append(d)
    
train_file = []

used_dict = {}
for data in tqdm(dataset_tmp_dict):
    good_sample_permutated = itertools.combinations(dataset_tmp_dict[data], 2)
    good_sample_list = []
    for gg in good_sample_permutated:
        good_sample_list.append([gg[0]['template'], gg[1]['template'], 1])

    rest_list = []
    for dd in dataset_tmp_dict:
        if dd == data:
            continue
        else:
            rest_list += dataset_tmp_dict[dd]
    bad_sample_list = []
    for dd in dataset_tmp_dict[data]:
        similarity = 1
        while similarity >= 0.3:
            source_embedding = dd['embedding']
            selected_target = random.sample(rest_list, 1)[0]
            target_text = selected_target['template']
            target_embedding = selected_target['embedding']
            
            if target_text in used_dict and used_dict[target_text] >= 4:
                continue
            
            similarity = float(torch.cosine_similarity(source_embedding.reshape(1,-1), target_embedding.reshape(1,-1)))
            
            if similarity < 0.3:
                if target_text not in used_dict:
                    used_dict[target_text] = 1
                else:
                    used_dict[target_text] += 1
                bad_sample_list.append([dd['template'], target_text, 0])

    train_file += random.sample(good_sample_list, int(len(good_sample_list)*0.4))
    train_file += bad_sample_list

print(len(train_file))

with open("data/train.json", "w", encoding='utf-8') as outfile:
    json.dump(train_file, outfile, indent=4)
