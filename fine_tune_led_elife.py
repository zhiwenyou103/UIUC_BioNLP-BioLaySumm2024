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
import random
from datasets import load_dataset, load_metric

random.seed(42)

# base
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

# large
# tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")

# 8192
encoder_max_length = 4096
decoder_max_length = 512
batch_size = 4


### process the sections headings file
# Initialize lists for each category
background = []
objective = []
methods = []
results = []
conclusions = []

with open('/ocean/projects/cis230089p/zyou2/Structured-Abstracts-Labels-102615.txt', 'r') as file:
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
    
print(background)
print(conclusions)

def process_data_to_model_inputs(batch):
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

from rouge import Rouge
rouge = Rouge()
# rouge = load_metric("rouge")
# the generation output, called pred.predictions as well as the gold label, called pred.label_ids.
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.get_scores(pred_str, label_str)[0]['rouge-2']
    
    # rouge_output = rouge.compute(
    #     predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    # )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output['p'], 4),
        "rouge2_recall": round(rouge_output['r'], 4),
        "rouge2_fmeasure": round(rouge_output['f'], 4),
    }


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

def load_test_data(dataset, datatype):

    data_folder = '/ocean/projects/cis230089p/zyou2/biolaysumm2024_data'
    
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

import json

### PLOS
# train
# plos_article_train, plos_lay_sum_train, plos_keyword_train, plos_headings_train, plos_id_train = load_data('PLOS', 'train')
# # val
# plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')
# # test
# plos_article_test, plos_keyword_test, plos_headings_test, plos_id_test = load_test_data('PLOS', 'test')


### eLife
# train
elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('eLife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')
# test
# elife_article_test, elife_keyword_test, elife_headings_test, elife_id_test = load_test_data('eLife', 'test')

### train
new_elife_article_train = []
new_elife_lay_sum_train = []

for lay, article, headings in zip(elife_lay_sum_train, elife_article_train, elife_headings_train):
    sections = article.split('\n')
    temp_sections = []
    for i, (heading, section) in enumerate(zip(headings, sections)):
        heading = heading.lower()
        if heading in background:
            temp_sections.append(section)
        elif heading in conclusions:
            temp_sections.append(section)
        elif heading in 'abstract':
            temp_sections.append(section)
        else:
            continue
    final_string = '\n'.join(temp_sections)
    if final_string:
        new_elife_article_train.append(final_string)
        new_elife_lay_sum_train.append(lay)

### val
new_elife_article_val = []
new_elife_lay_sum_val = []

for lay, article, headings in zip(elife_lay_sum_val, elife_article_val, elife_headings_val):
    sections = article.split('\n')
    temp_sections = []
    for i, (heading, section) in enumerate(zip(headings, sections)):
        heading = heading.lower()
        if heading in background:
            temp_sections.append(section)
        elif heading in conclusions:
            temp_sections.append(section)
        elif heading in 'abstract':
            temp_sections.append(section)
        else:
            continue
    final_string = '\n'.join(temp_sections)
    if final_string:
        new_elife_article_val.append(final_string)
        new_elife_lay_sum_val.append(lay)



# plos train
elife_train_dataset = {'article': new_elife_article_train, 'abstract': new_elife_lay_sum_train}
elife_train_dataset = Dataset.from_dict(elife_train_dataset)

# val
elife_val_dataset = {'article': new_elife_article_val, 'abstract': new_elife_lay_sum_val}
elife_val_dataset = Dataset.from_dict(elife_val_dataset)

# plos_train_dataset = plos_train_dataset.select(range(100))
# plos_val_dataset = plos_val_dataset.select(range(100))


print("length of train and val datasets")
print(len(elife_train_dataset))
print(len(elife_val_dataset))


# map train data
train_dataset = elife_train_dataset.map(
    process_data_to_model_inputs,
    batched = True,
    batch_size = batch_size,
    remove_columns=["article", "abstract"]
)
# map val data
val_dataset = elife_val_dataset.map(
    process_data_to_model_inputs,
    batched = True,
    batch_size = batch_size,
    remove_columns=["article", "abstract"]
)

# the datasets should be converted into the PyTorch format
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)


led_model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)

# set generate hyperparameters
led_model.config.num_beams = 2
led_model.config.max_length = 512
led_model.config.min_length = 100
led_model.config.length_penalty = 2.0
led_model.config.early_stopping = True
led_model.config.no_repeat_ngram_size = 3

# Training
# model_name = 'elife_8k_3sec'
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    output_dir="/ocean/projects/cis230089p/zyou2/BioNLP/4k_led_base",
    logging_steps=5,
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    load_best_model_at_end=True,
)

print("start training...")

trainer = Seq2SeqTrainer(
    model=led_model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# trainer.evaluate()