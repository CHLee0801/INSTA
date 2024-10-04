from sentence_transformers import SentenceTransformer, InputExample
import json
from sklearn.metrics.pairwise import paired_cosine_distances
import sys

train_dict = {
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

    'multi_news' : ['summarization', 'multi_news', 'validation', 0],
    'gigaword' : ['summarization', 'gigaword', 'validation', 0],
    'samsum' : ['summarization', 'samsum', 'validation', 0],
    'xsum' : ['summarization', 'xsum', 'validation', 0],
    'cnn_daily_mail' : ['summarization', 'cnn_dailymail/3.0.0', 'validation', 0],

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
    
    'common_gen' : ['struct_to_text', 'common_gen', 'validation', 0],
    'wiki_bio' : ['struct_to_text', 'wiki_bio', 'val', 0]
}

eval_dict = { 
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

if sys.argv[1] == '0':
    eval_type = 'by_dataset' 
elif sys.argv[1] == '1':
    eval_type = 'by_dataset_task_zero'
elif sys.argv[1] == '2':
    eval_type = 'by_task'

model = SentenceTransformer(f'/home/changho.lee/MT_Selector/selector_model/ckpt/updated/343').cuda()

train_path = '/home/changho.lee/MT_Selector/template_selection/train_preprocessed.json'
eval_path = '/home/changho.lee/MT_Selector/template_selection/test_preprocessed.json'

with open(train_path, 'r') as file:
    train_data = json.load(file)
    
with open(eval_path, 'r') as file:
    eval_data = json.load(file)
sentence_1, sentence_2 = [], []

out_list = []
for tt in train_data:
    if tt['dataset'] not in train_dict:
        continue
    for ee in eval_data:
        sentence_1.append(tt['template'])
        sentence_2.append(ee['template'])
        out_list.append([tt['dataset'], ee['dataset']])
        
embeddings1 = model.encode(sentence_1, batch_size=4096, show_progress_bar=True,convert_to_numpy=True)
embeddings2 = model.encode(sentence_2, batch_size=4096, show_progress_bar=True,convert_to_numpy=True)
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

for idx, score in enumerate(cosine_scores):
    out_list[idx].append(cosine_scores[idx])

evaluation_cluster = ['natural_language_inference', 'word_sense_disambiguation', 'coreference_resolution', 'sentence_completion']

out_dict = {}
for instance in out_list:
    if instance[1] not in out_dict:
        out_dict[instance[1]] = {}
    if instance[0] not in out_dict[instance[1]]:
        out_dict[instance[1]][instance[0]] = [instance[2], 1, instance[2]]
    else:
        out_dict[instance[1]][instance[0]][0] += instance[2]
        out_dict[instance[1]][instance[0]][1] += 1
        out_dict[instance[1]][instance[0]][2] = max(out_dict[instance[1]][instance[0]][2], instance[2])
        
if eval_type == 'by_dataset' or eval_type == 'by_dataset_task_zero' or eval_type == 'by_task':
    for out in out_dict:
        #if out != '':
        #    continue
        print(out)
        sorting_list = []

        for oo in out_dict[out]:
            sorting_list.append([oo, out_dict[out][oo][0] / out_dict[out][oo][1], out_dict[out][oo][2]])
        sorting_list = sorted(sorting_list, key=lambda x: x[1], reverse=True)

        for i in range(5):
            print(sorting_list[i])
        print("===================================")
elif eval_type == 'by_task':
    for out in out_dict:
        for oo in out_dict[out]:
            out_dict[out][oo].append(out_dict[out][oo][0] / out_dict[out][oo][1])
    
    task_out = {
        'natural_language_inference':{},
        'word_sense_disambiguation':{},
        'coreference_resolution':{},
        'sentence_completion':{}
    }
    for out in out_dict:
        for oo in out_dict[out]:
            print(out)
            print(oo)
            if train_dict[oo][0] not in task_out[eval_dict[out][0]]:
                task_out[eval_dict[out][0]][train_dict[oo][0]] = [out_dict[out][oo][2],1]
            else:
                task_out[eval_dict[out][0]][train_dict[oo][0]][0] += out_dict[out][oo][2]
                task_out[eval_dict[out][0]][train_dict[oo][0]][1] += 1
    
    for out in task_out:
        print(out)
        final_sorting_list = []
        for oo in task_out[out]:
            final_sorting_list.append([oo, task_out[out][oo][0] / task_out[out][oo][1]])
        final_sorting_list = sorted(final_sorting_list, key=lambda x:x[1], reverse=True)
        for ff in final_sorting_list:
            print(ff)
        print("===================================")
