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
import numpy as np
import nltk
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from transformers import AutoTokenizer, LongT5Model
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

random.seed(42)
# nltk.download('punkt')

metric = load_metric("rouge")
# 
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")

# tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
# label_pad_token_id = tokenizer.pad_token_id
model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
# AutoModelForSeq2SeqLM LongT5ForConditionalGeneration
# 16384 8192
encoder_max_length = 16384
decoder_max_length = 512
batch_size = 1

def process_data_to_model_inputs(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=encoder_max_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["abstract"], max_length=decoder_max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# def process_data_to_model_inputs(batch):
#     inputs = tokenizer(
#         batch["article"],
#         padding="max_length",
#         truncation=True,
#         max_length=encoder_max_length,
#     )
#     outputs = tokenizer(
#         batch["abstract"],
#         padding="max_length",
#         truncation=True,
#         max_length=decoder_max_length,
#     )
#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     # batch["global_attention_mask"] = len(batch["input_ids"]) * [
#     #     [0 for _ in range(len(batch["input_ids"][0]))]
#     # ]

#     # batch["global_attention_mask"][0][0] = 1
#     batch["labels"] = outputs.input_ids

#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels]
#         for labels in batch["labels"]
#     ]

#     return batch

from rouge import Rouge
rouge = Rouge()
# rouge = load_metric("rouge")
# the generation output, called pred.predictions as well as the gold label, called pred.label_ids.
# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
#     rouge_output = rouge.get_scores(pred_str, label_str)[0]['rouge-2']
    
#     # rouge_output = rouge.compute(
#     #     predictions=pred_str, references=label_str, rouge_types=["rouge2"]
#     # )["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output['p'], 4),
#         "rouge2_recall": round(rouge_output['r'], 4),
#         "rouge2_fmeasure": round(rouge_output['f'], 4),
#     }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    
    # added to overcome overflow
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


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

# def load_test_data(dataset, datatype):

#     data_folder = '/ocean/projects/cis230089p/zyou2/biolaysumm2024_data'
    
#     data_path = os.path.join(data_folder, f'{dataset}_{datatype}.jsonl')
#     article =[]
#     keyword = []
#     headings = []
#     id = []
#     file = open(data_path, 'r')
#     for line in (file.readlines()):
#         dic = json.loads(line)
#         article.append(dic['article'])
#         keyword.append(dic['keywords'])
#         headings.append(dic['headings'])
#         id.append(dic['id'])
    
#     return article, keyword, headings, id

# import json

# ### PLOS
# # train
# # plos_article_train, plos_lay_sum_train, plos_keyword_train, plos_headings_train, plos_id_train = load_data('PLOS', 'train')
# # # val
# # plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')
# # # test
# # plos_article_test, plos_keyword_test, plos_headings_test, plos_id_test = load_test_data('PLOS', 'test')


### eLife
# train
elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('eLife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')
# test
# elife_article_test, elife_keyword_test, elife_headings_test, elife_id_test = load_test_data('eLife', 'test')


# elife train
elife_train_dataset = {'article': elife_article_train, 'abstract': elife_lay_sum_train}
elife_train_dataset = Dataset.from_dict(elife_train_dataset)

# val
elife_val_dataset = {'article': elife_article_val, 'abstract': elife_lay_sum_val}
elife_val_dataset = Dataset.from_dict(elife_val_dataset)

print("length of train and val datasets")
print(len(elife_train_dataset))
print(len(elife_val_dataset))


# map train data
# train_dataset = elife_train_dataset.map(
#     process_data_to_model_inputs,
#     batched = True,
#     batch_size = batch_size,
#     remove_columns=["article", "abstract"]
# )
train_dataset = elife_train_dataset.map(
    process_data_to_model_inputs,
    batched = True
)
val_dataset = elife_val_dataset.map(
    process_data_to_model_inputs,
    batched = True
)
# map val data
# val_dataset = elife_val_dataset.map(
#     process_data_to_model_inputs,
#     batched = True,
#     batch_size = batch_size,
#     remove_columns=["article", "abstract"]
# )

# # the datasets should be converted into the PyTorch format
# train_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )
# val_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )

# train_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "labels"],
# )
# val_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "labels"],
# )


# Training
    # eval_steps=2000,
    # save_steps=4000,
# generation_max_length=512,

model.config.num_beams = 2
model.config.max_length = 512
# model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

model_name = 'longt5_elife'
# training_args = Seq2SeqTrainingArguments(
#     predict_with_generate=True,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     fp16=False,
#     output_dir=f"/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/{model_name}",
#     logging_steps=4346,
#     save_total_limit=3,
#     learning_rate=0.001,
#     weight_decay=0.01,
#     num_train_epochs=8,
#     load_best_model_at_end=True,
# )
#save_strategy="epoch",
# logging_steps=5,
#    eval_steps=500,save_steps=50,
training_args = Seq2SeqTrainingArguments(
    output_dir=f"/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/16384_base/{model_name}",
    evaluation_strategy = "steps",
    save_strategy="steps",
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_steps=600,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False, # change to fp16 in no Ampere GPU  available.
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    adafactor=True,
)

print("start training...")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=None)

# LoRA
model.config.use_cache = False
model.enable_input_require_grads()
lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.base_model.model.encoder.enable_input_require_grads()
model.base_model.model.decoder.enable_input_require_grads()


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.model.save_pretrained(os.path.join(f"/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/16384_base/{model_name}", 'checkpoint-best'))
model.base_model.save_pretrained(os.path.join(f"/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/16384_base/{model_name}", 'checkpoint-base-model'))

# from peft import PeftModelForSeq2SeqLM
# # original_model = AutoModelForSeq2SeqLM.from_pretrained(
# #   model.config["google/long-t5-tglobal-base"]
# # )
# original_with_adapter = PeftModelForSeq2SeqLM.from_pretrained(
#   model, f"/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/{model_name}/checkpoint-best" # bert-peft; the folder of the saved adapter
# )
# merged_model = original_with_adapter.merge_and_unload()
# merged_model.save_pretrained("/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/merged-model")

print("finished training")
# result = trainer.predict(test_dataset=val_dataset)
# decoded_preds = tokenizer.batch_decode(result.predictions, skip_special_tokens=True)

# with open('/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/longt5_val_predicted_abstracts.txt', 'w') as file:
#     for abstract in decoded_preds:
#         file.write(abstract + '\n')
        
# print("finished writing")
# trainer.evaluate()