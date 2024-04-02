
from transformers import(
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from tqdm import tqdm
from datasets import Dataset
import os
import json
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import math
import random
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import torch
import re
from rouge import Rouge
from transformers import AutoTokenizer
from summarizer.sbert import SBertSummarizer
from summarizer import Summarizer
from transformers import *
import nltk
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import re
from langchain_text_splitters import NLTKTextSplitter

import random
import numpy as np
# set random seed
random_seed = 42
random.seed(random_seed)

wiki_path = '/home/zhiweny2/chatbotai/jerome/BioNLP/wiki/wiki_dict.json'
wiki = json.load(open(wiki_path))

def load_data(dataset, datatype):

    data_folder = '/home/zhiweny2/chatbotai/jerome/BioNLP/biolaysumm2024_data'
    data_path = os.path.join(data_folder, f'{dataset}_{datatype}.jsonl')
    lay_sum = []
    article =[]
    keyword = []
    headings = []
    id = []
    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        article.append(dic['article'])
        keyword.append(dic['keywords'])
        headings.append(dic['headings'])
        id.append(dic['id'])
        lay_sum.append(dic['lay_summary'])
    
    return article, lay_sum, keyword, headings, id

def load_test_data(dataset, datatype):

    data_folder = '/home/zhiweny2/chatbotai/jerome/BioNLP/biolaysumm2024_data'
    
    data_path = os.path.join(data_folder, f'{dataset}_{datatype}.jsonl')
    article =[]
    keyword = []
    headings = []
    id = []
    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        article.append(dic['article'])
        keyword.append(dic['keywords'])
        headings.append(dic['headings'])
        id.append(dic['id'])
    
    return article, keyword, headings, id

background = []
objective = []
methods = []
results = []
conclusions = []

with open('/home/zhiweny2/chatbotai/jerome/BioNLP/Structured-Abstracts-Labels-102615.txt', 'r') as file:
    for line in file:
        components = line.strip().split('|')
        title, category, _, _ = components
        if category == 'BACKGROUND':
            background.append(title)
        elif category == 'OBJECTIVE':
            objective.append(title)
        elif category == 'METHODS':
            methods.append(title)
        elif category == 'RESULTS':
            results.append(title)
        elif category == 'CONCLUSIONS':
            conclusions.append(title)
            
background = [item.lower() for item in background]
objective = [item.lower() for item in objective]
methods = [item.lower() for item in methods]
results = [item.lower() for item in results]
conclusions = [item.lower() for item in conclusions]

### PLOS
# plos_article_train, plos_lay_sum_train, plos_keyword_train, plos_headings_train, plos_id_train = load_data('PLOS', 'train')
# # val
# plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')

elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('eLife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')

### chunk articles and lay summs

### chunk articles and lay summs for PLOS 

# pattern = r'\s\[.*?\]' # PLOS
# pattern = r'(\(\s([^()]*\s\,\s)*[^()]*\s\))' # elife
text_splitter = NLTKTextSplitter(chunk_size=600)

### train
new_plos_article_train = []
for s in elife_article_train:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_article_train.append(new_s)
print(len(new_plos_article_train))
    
### val
new_plos_article_val = []
for s in elife_article_val:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_article_val.append(new_s)
print(len(new_plos_article_val))

### train lay sum
new_plos_lay_sum_train = []
for s in elife_lay_sum_train:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_lay_sum_train.append(new_s)

### val lay sum
new_plos_lay_sum_val = []
for s in elife_lay_sum_val:
    new_s = s.replace(' . ', '. ')
    new_s = new_s.replace(' , ', ', ')
    new_plos_lay_sum_val.append(new_s)


### create new extractive datasets
# text_splitter = NLTKTextSplitter(chunk_size=600)

custom_config = AutoConfig.from_pretrained('/home/zhiweny2/chatbotai/jerome/BioNLP/elife_sentence_level_embedding_model')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('/home/zhiweny2/chatbotai/jerome/BioNLP/elife_sentence_level_embedding_model')
custom_model = AutoModel.from_pretrained('/home/zhiweny2/chatbotai/jerome/BioNLP/elife_sentence_level_embedding_model', config=custom_config)
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)


non_selected_sec_train = []
selected_sec_train = []
for lay, article, headings, keywords in zip(new_plos_lay_sum_train, new_plos_article_train, elife_headings_train, elife_keyword_train):
    sections = article.split('\n')
    temp_selected_sections = []
    temp_sections = []
    for i, (heading, section) in enumerate(zip(headings, sections)):
        heading = heading.lower()
        if heading in background:
            temp_selected_sections.append(section)
        elif heading in methods:
            temp_sections.append(section)
        elif heading in conclusions:
            temp_selected_sections.append(section)
        elif heading in results:
            temp_sections.append(section)
        elif heading in 'abstract':
            temp_selected_sections.append(section)
        else:
            temp_sections.append(section)
            # continue
    
    for keyword in keywords:
        if keyword in wiki:
            tmp = wiki[keyword]
            total = ""
            for i in range(len(tmp)):
                if type(tmp[i]) == str:
                    total += keyword + ' is ' + tmp[i] + '. '
            if total:
                temp_selected_sections.append(total)
        else:
            continue
    
    final_string = ''.join(temp_sections)
    final_selected_string = ''.join(temp_selected_sections)
    non_selected_sec_train.append(final_string)
    selected_sec_train.append(final_selected_string)
    
print("non selected sections train")
print(len(non_selected_sec_train))
print(len(selected_sec_train))

chunked_plos_article_train = []
for article in non_selected_sec_train:
    texts = text_splitter.split_text(article)
    new_texts = []
    # elife don't remove any citations but PLOS does
    for t in texts:
        t = t.replace('\n\n', ' ')
        # result = re.sub(pattern, "", t)
        result = t.replace(' , ', ', ')
        new_texts.append(result)
    chunked_string = ' '.join(new_texts)
    chunked_plos_article_train.append(chunked_string)
    
print("chunked plos article train")
print(len(chunked_plos_article_train))

non_selected_sec_val = []
selected_sec_val = []
for lay, article, headings, keywords in zip(new_plos_lay_sum_val, new_plos_article_val, elife_headings_val, elife_keyword_val):
    sections = article.split('\n')
    temp_selected_sections = []
    temp_sections = []
    for i, (heading, section) in enumerate(zip(headings, sections)):
        heading = heading.lower()
        if heading in background:
            temp_selected_sections.append(section)
        elif heading in methods:
            temp_sections.append(section)
        elif heading in conclusions:
            temp_selected_sections.append(section)
        elif heading in results:
            temp_sections.append(section)
        elif heading in 'abstract':
            temp_selected_sections.append(section)
        else:
            temp_sections.append(section)
            # continue
    
    for keyword in keywords:
        if keyword in wiki:
            tmp = wiki[keyword]
            total = ""
            for i in range(len(tmp)):
                if type(tmp[i]) == str:
                    total += keyword + ' is ' + tmp[i] + '. '
            if total:
                temp_selected_sections.append(total)
        else:
            continue
       
    final_string = ''.join(temp_sections)
    final_selected_string = ''.join(temp_selected_sections)
    non_selected_sec_val.append(final_string)
    selected_sec_val.append(final_selected_string)
    
print("non selected sections val")
print(len(non_selected_sec_val))
print(len(selected_sec_val))

chunked_plos_article_val = []
for article in non_selected_sec_val:
    texts = text_splitter.split_text(article)
    new_texts = []
    for t in texts:
        t = t.replace('\n\n', ' ')
        ### remove the irrelevant citations and references
        # result = re.sub(pattern, "", t)
        result = t.replace(' , ', ', ')
        new_texts.append(result)
    chunked_string = ' '.join(new_texts)
    chunked_plos_article_val.append(chunked_string)
    
print("chunked plos article val")
print(len(chunked_plos_article_val))

### train
final_elife_lay_sum_train = []
new_selected_elife_article_train = []
extractive_sum_train = []
for lay, article, select in zip(new_plos_lay_sum_train, chunked_plos_article_train, selected_sec_train):
    summarized_sections = ""
    if select:
        final_elife_lay_sum_train.append(lay)
        result = model(article, num_sentences=50, max_length=256)
        summarized_sections = ''.join(result)
        extractive_sum_train.append(summarized_sections)
        # PLOS
        # new_select = re.sub(pattern, '', select)
        # new_select = new_select.replace(' , ', ', ')
        # elife
        new_select = select.replace(' , ', ', ')
        new_select = new_select + '\n' + summarized_sections
        new_selected_elife_article_train.append(new_select)
    else:
        new_selected_elife_article_train.append(select)
print(len(new_selected_elife_article_train))

### val
final_elife_lay_sum_val = []
new_selected_elife_article_val = []
extractive_sum_val = []
for lay, article, select in zip(new_plos_lay_sum_val, chunked_plos_article_val, selected_sec_val):
    summarized_sections = ""
    if select:
        final_elife_lay_sum_val.append(lay)
        result = model(article, num_sentences=50, max_length=256)
        summarized_sections = ''.join(result)
        extractive_sum_val.append(summarized_sections)
        # PLOS
        # new_select = re.sub(pattern, '', select)
        # new_select = new_select.replace(' , ', ', ')
        # elife
        new_select = select.replace(' , ', ', ')
        new_select = new_select + '\n' + summarized_sections
        new_selected_elife_article_val.append(new_select)
    else:
        new_selected_elife_article_val.append(select)
        
print(len(new_selected_elife_article_val))
print(len(extractive_sum_val))

output_file = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/elife_train.json"
with open(output_file, "w") as file:
    json.dump(new_selected_elife_article_train, file)
    
output_file_1 = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/elife_val.json"
with open(output_file_1, "w") as file:
    json.dump(new_selected_elife_article_val, file)
    
output_file = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/elife_train_extractive_sum.json"
with open(output_file, "w") as file:
    json.dump(extractive_sum_train, file)
    
output_file_1 = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/elife_val_extractive_sum.json"
with open(output_file_1, "w") as file:
    json.dump(extractive_sum_val, file)  















### train
# final_elife_lay_sum_train = []
# new_selected_elife_article_train = []
# extractive_sum_train = []
# for lay, article, headings in zip(new_plos_lay_sum_train, new_plos_article_train, plos_headings_train):
#     sections = article.split('\n')
#     temp_selected_sections = []
#     temp_sections = []
#     for i, (heading, section) in enumerate(zip(headings, sections)):
#         heading = heading.lower()
#         if heading in background:
#             temp_selected_sections.append(section)
#         elif heading in methods:
#             temp_sections.append(section)
#         elif heading in conclusions:
#             temp_selected_sections.append(section)
#         elif heading in results:
#             temp_sections.append(section)
#         elif heading in 'abstract':
#             temp_selected_sections.append(section)
#         else:
#             continue
        
#     final_string = '\n'.join(temp_sections)
#     final_selected_string = '\n'.join(temp_selected_sections)

#     chunked_string = ""
#     summarized_sections = ""
#     if final_selected_string:
#         final_elife_lay_sum_train.append(lay)
#         if final_string:
#             a = text_splitter.split_text(final_string)
#             new_texts = []
#             for t in a:
#                 t = t.replace('\n\n', ' ')
#                 ### remove the irrelevant citations and references
#                 result = re.sub(pattern, "", t)
#                 result = result.replace(' , ', ', ')
#                 new_texts.append(result)
#             chunked_string = ' '.join(new_texts)

#         if chunked_string:
#             result = model(chunked_string, num_sentences=10, max_length=256) # num=50 for elife
#             summarized_sections = ''.join(result)
#         if summarized_sections:
#             extractive_sum_train.append(summarized_sections)
#             final_selected_string = re.sub(pattern, '', final_selected_string)
#             final_selected_string = final_selected_string.replace(' , ', ', ')
#             final_selected_string = final_selected_string + '\n' + summarized_sections
#             new_selected_elife_article_train.append(final_selected_string)
#         else:
#             new_selected_elife_article_train.append(final_selected_string)
            
# print(len(new_selected_elife_article_train))


# ### val
# final_elife_lay_sum_val = []
# new_selected_elife_article_val = []
# extractive_sum_val = []
# for lay, article, headings in zip(new_plos_lay_sum_val, new_plos_article_val, plos_headings_val):
#     sections = article.split('\n')
#     temp_selected_sections = []
#     temp_sections = []
#     for i, (heading, section) in enumerate(zip(headings, sections)):
#         heading = heading.lower()
#         if heading in background:
#             temp_selected_sections.append(section)
#         elif heading in methods:
#             temp_sections.append(section)
#         elif heading in conclusions:
#             temp_selected_sections.append(section)
#         elif heading in results:
#             temp_sections.append(section)
#         elif heading in 'abstract':
#             temp_selected_sections.append(section)
#         else:
#             continue
        
#     final_string = '\n'.join(temp_sections)
#     final_selected_string = '\n'.join(temp_selected_sections)

#     chunked_string = ""
#     summarized_sections = ""
#     if final_selected_string:
#         final_elife_lay_sum_val.append(lay)
    
#         if final_string:
#             a = text_splitter.split_text(final_string)
#             new_texts = []
#             for t in a:
#                 t = t.replace('\n\n', ' ')
#                 ### remove the irrelevant citations and references
#                 result = re.sub(pattern, "", t)
#                 result = result.replace(' , ', ', ')
#                 new_texts.append(result)
#             chunked_string = ' '.join(new_texts)

#         if chunked_string:
#             result = model(chunked_string, num_sentences=10, max_length=256)
#             summarized_sections = ''.join(result)
#         if summarized_sections:
#             extractive_sum_val.append(summarized_sections)
#             final_selected_string = re.sub(pattern, '', final_selected_string)
#             final_selected_string = final_selected_string.replace(' , ', ', ')
#             final_selected_string = final_selected_string + '\n' + summarized_sections
#             new_selected_elife_article_val.append(final_selected_string)
#         else:
#             new_selected_elife_article_val.append(final_selected_string)
            
# print(len(new_selected_elife_article_val))
# print(len(extractive_sum_val))

# output_file = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/plos_train.json"
# with open(output_file, "w") as file:
#     json.dump(new_selected_elife_article_train, file)
    
# output_file_1 = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/plos_val.json"
# with open(output_file_1, "w") as file:
#     json.dump(new_selected_elife_article_val, file)
    
# output_file = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/plos_train_extractive_sum.json"
# with open(output_file, "w") as file:
#     json.dump(extractive_sum_train, file)
    
# output_file_1 = "/home/zhiweny2/chatbotai/jerome/BioNLP/extractive_aug_dataset/plos_val_extractive_sum.json"
# with open(output_file_1, "w") as file:
#     json.dump(extractive_sum_val, file)