from sentence_transformers import SentenceTransformer, util
import torch
import json
from utils import *

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


background = []
objective = []
methods = []
results = []
conclusions = []

# Open and read the .txt file
with open('/jet/home/zyou2/BioLaySumm/Structured-Abstracts-Labels-102615.txt', 'r') as file:
    for line in file:
        # Split each line into components
        components = line.strip().split('|')
        
        # Extract the first column (title) and second column (category)
        title, category, _, _ = components
        
        # Categorize the title based on the category
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

print(len(background), len(objective), len(methods), len(results), len(conclusions))


def load_data(dataset, datatype):

    data_folder = '/ocean/projects/cis230089p/zyou2/biolaysumm2024_data'
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

### PLOS
# train
# plos_article_train, plos_lay_sum_train, plos_keyword_train, plos_headings_train, plos_id_train = load_data('PLOS', 'train')
# # val
# plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')

### eLife
elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('eLife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')

abstract_scores = []
background_scores = []
objective_scores = []
methods_scores = []
results_scores = []
conclusion_scores = []

model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_cos_sim(lay_embedding, section_embeddings):
    cosine_scores = util.cos_sim(lay_embedding, section_embeddings)
    return cosine_scores

model = model.to(torch.device("cuda"))

lay_embeddings = model.encode(elife_lay_sum_train, convert_to_tensor=True, batch_size=32)
section_embeddings_list = [model.encode(article.split('\n'), convert_to_tensor=True, batch_size=32) for article in elife_article_train]

print("lay embeddings length")
print(len(lay_embeddings))
print(len(section_embeddings_list))

count = 0
for lay_embedding, section_embeddings, headings in zip(lay_embeddings, section_embeddings_list, elife_headings_train):
    cosine_scores = calculate_cos_sim(lay_embedding, section_embeddings).cpu().numpy()
    for i, heading in enumerate(headings):
        heading = heading.lower()
        score = "{:.4f}".format(cosine_scores[0][i])
        if heading in background:
            background_scores.append(score)
        elif heading in objective:
            objective_scores.append(score)
        elif heading in methods:
            methods_scores.append(score)
        elif heading in results:
            results_scores.append(score)
        elif heading in conclusions:
            conclusion_scores.append(score)
        elif heading in 'abstract':
            abstract_scores.append(score)
        else:
            continue
    # print("finish one article calculation")
    count += 1
    
print(count)
print(len(abstract_scores))
print(len(background_scores))
print(len(objective_scores))
print(len(methods_scores))
print(len(results_scores))
print(len(conclusion_scores))


with open('/jet/home/zyou2/BioLaySumm/elife_similarity/abstract_scores.txt', 'w') as file:
    file.writelines(score + '\n' for score in abstract_scores)

with open('/jet/home/zyou2/BioLaySumm/elife_similarity/background_scores.txt', 'w') as file1:
    file1.writelines(score + '\n' for score in background_scores)

with open('/jet/home/zyou2/BioLaySumm/elife_similarity/objective_scores.txt', 'w') as file2:
    file2.writelines(score + '\n' for score in objective_scores)

with open('/jet/home/zyou2/BioLaySumm/elife_similarity/methods_scores.txt', 'w') as file3:
    file3.writelines(score + '\n' for score in methods_scores)

with open('/jet/home/zyou2/BioLaySumm/elife_similarity/results_scores.txt', 'w') as file4:
    file4.writelines(score + '\n' for score in results_scores)

with open('/jet/home/zyou2/BioLaySumm/elife_similarity/conclusion_scores.txt', 'w') as file5:
    file5.writelines(score + '\n' for score in conclusion_scores)