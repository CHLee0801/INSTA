import json
import itertools
import random

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
    'hellaswag' : ['sentence_completion', 'hellaswag', 'validation', 1],
    
    'boolq' : ['multiple_choice_qa', 'super_glue/boolq', 'validation', 1],
    'multirc' : ['multiple_choice_qa', 'super_glue/multirc', 'validation', 1],
    'record' : ['multiple_choice_qa', 'super_glue/record', 'validation', 0],
    'race' : ['multiple_choice_qa', 'race/all', 'validation', 0],
    'squad' : ['extractive_qa', 'squad', 'validation', 0],
    'drop' : ['extractive_qa', 'drop', 'validation', 0],
    'arc' : ['multiple_choice_qa', 'ai2_arc/ARC-Challenge', 'validation', 1],
    'piqa' : ['multiple_choice_qa', 'piqa', 'validation', 1],
    'openbookqa' : ['multiple_choice_qa', 'openbookqa/main', 'validation', 1],
    'cbt' : ['multiple_choice_qa', 'cbt/CN', 'validation', 1],
    'art' : ['multiple_choice_qa', 'art', 'validation', 1]
}

with open('data/train_instances.json', 'r') as file:
    train_data = json.load(file)



with open('data/dev_instances.json', 'r') as file:
    data = json.load(file)

dataset_tmp_dict = {}

for i in data:
    if i['dataset'] not in dataset_tmp_dict:
        dataset_tmp_dict[i['dataset']] = []
    dataset_tmp_dict[i['dataset']].append(i['template'])
    
dev_file = []

for data in dataset_tmp_dict:
    good_sample_permutated = itertools.combinations(dataset_tmp_dict[data], 2)
    good_sample_list = []
    for gg in good_sample_permutated:
        good_sample_list.append([gg[0], gg[1], 1])

    rest_list = []
    for dd in dataset_tmp_dict:
        if dd == dataset_dict[data][0]:
            continue
        else:
            rest_list += dataset_tmp_dict[dd]
    bad_sample_list = []
    for dd in dataset_tmp_dict[data]:
        bad_sample_list.append([dd, random.sample(rest_list, 1)[0], 0])
    dev_file += good_sample_list
    dev_file += bad_sample_list

with open("data/dev.json", "w", encoding='utf-8') as outfile:
    json.dump(dev_file, outfile, indent=4)
