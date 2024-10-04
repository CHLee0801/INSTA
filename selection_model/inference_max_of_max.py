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
    'wiki_hop' : ['multiple_choice_qa', 'wiki_hop/original', 'validation', 1]
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

#eval_type = 'by_dataset' 
ckpt_path = sys.argv[1]
model = SentenceTransformer(f'/home/ubuntu/ex_disk/MT_Selector/selection_model/ckpt/{ckpt_path}').cuda() 

train_path = '/home/ubuntu/ex_disk/MT_Selector/selection_model/data/more_filter_1130/train_preprocessed.json'
eval_path = '/home/ubuntu/ex_disk/MT_Selector/selection_model/data/more_filter_1130/test_preprocessed.json'

with open(train_path, 'r') as file:
    train_data = json.load(file)
    
with open(eval_path, 'r') as file:
    eval_data = json.load(file)
sentence_1, sentence_2 = [], []

out_list = []
cnt = 0
for tt in train_data:
    if tt['dataset'] not in train_dict:
        continue
    cnt += 1
    for ee in eval_data:
        sentence_1.append(tt['template'])
        sentence_2.append(ee['template'])
        out_list.append([tt['dataset'], ee['dataset'], ee['template'], tt['template']])

embeddings1 = model.encode(sentence_1, batch_size=1024, show_progress_bar=True,convert_to_numpy=True)
embeddings2 = model.encode(sentence_2, batch_size=1024, show_progress_bar=True,convert_to_numpy=True)
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

for idx, score in enumerate(cosine_scores):
    out_list[idx].append(cosine_scores[idx])

evaluation_cluster = ['natural_language_inference', 'word_sense_disambiguation', 'coreference_resolution', 'sentence_completion']

out_dict = {}
for instance in out_list:
    if instance[1] not in out_dict:
        out_dict[instance[1]] = {}
    if instance[2] not in out_dict[instance[1]]:
        out_dict[instance[1]][instance[2]] = []
    #if instance[0] not in out_dict[instance[1]][instance[2]]:
    #    out_dict[instance[1]][instance[2]][instance[0]] = []
    out_dict[instance[1]][instance[2]].append([instance[0], instance[3], instance[4]])
    

def voting(targetting_list):
    hey_dict = {}
    for i in range(len(targetting_list)):
        for j in range(10):
            if targetting_list[i][j] not in hey_dict:
                hey_dict[targetting_list[i][j]] = 0
            hey_dict[targetting_list[i][j]] += pow(10, 3*(10-j))
    ff_list = []
    for hey in hey_dict:
        ff_list.append([hey, hey_dict[hey]])
    ff_list = sorted(ff_list, key=lambda x:x[1], reverse=True)
    for i in range(5):
        print(ff_list[i][0])
def voting_2(targetting_list):
    hey_dict = {}
    print("==============")
    for i in range(len(targetting_list)):
        for j in range(10):
            if targetting_list[i][j][0] not in hey_dict:
                hey_dict[targetting_list[i][j][0]] = 0
            hey_dict[targetting_list[i][j][0]] += pow(10, 3*(10-j)) * targetting_list[i][j][1]
    ff_list = []
    for hey in hey_dict:
        ff_list.append([hey, hey_dict[hey]])
    ff_list = sorted(ff_list, key=lambda x:x[1], reverse=True)
    for i in range(5):
        print(ff_list[i][0])
        
for out in out_dict:
    #if out != '':
    #    continue
    print(out)  
    sorsor = []
    sorsorsor_list = []
    
    for ooo in out_dict[out]:
        print(ooo)
        sub_sorsor = []
        sub_sorsorsor = []
        sorting_list = sorted(out_dict[out][ooo], key=lambda x:x[2], reverse=True)
        for i in range(10):
            print(sorting_list[i])
            sub_sorsor.append(sorting_list[i][0])
            sub_sorsorsor.append([sorting_list[i][0], sorting_list[i][2]])
        print("===================================")
        sorsor.append(sub_sorsor)
        sorsorsor_list.append(sub_sorsorsor)
    print("TOP 5 RESULTS", out)
    
    voting(sorsor)
    voting_2(sorsorsor_list)