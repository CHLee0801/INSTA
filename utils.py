import string
import re
from rouge import Rouge
from collections import Counter
from konlpy.tag import Mecab
import datasets

from sklearn.metrics import recall_score, precision_score,f1_score,accuracy_score

def clean_up(text):
    text =text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text   

def clean(text):
    REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9가-힣]")
    return REMOVE_CHAR_PATTERN.sub(" ", text.lower()).strip()

def metric_rouge_english(preds, refs):
    metric = datasets.load_metric('rouge')
    
    results = metric.compute(predictions=preds, references=refs)
    
    #print(f"[ROUGE-L] {results['rougeL'].mid} \n[ROUGE-1] {results['rouge1'].mid} \n[ROUGE-2] {results['rouge2'].mid}")
    return results['rougeL'].mid.fmeasure

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return (white_space_fix(remove_punc(lower(s))))

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return lmap(str.strip, gen_text)

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def generation_metric(preds, refs, eval=False):
    rouge_score = metric_rouge_english(preds, refs)
    
    if eval==True:
        print(f"ROUGE-L Score : {rouge_score}")
        #accuracy = calculate_accuracy_scores(preds, refs)
        #print(f"Accuracy : {accuracy}")
        return rouge_score