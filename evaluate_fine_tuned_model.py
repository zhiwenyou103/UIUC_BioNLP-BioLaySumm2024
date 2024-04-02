import torch
from datasets import Dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration
from utils import *
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModelForSeq2SeqLM, PeftModel
from transformers import GenerationConfig
import random
import re


random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# pattern = r'(\(\s([^()]*\s\,\s)*[^()]*\s\))'
# pattern = r'\s\[.*?\]'

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


# load tokenizer
# longt5

# peft_model_id = "/jet/home/zyou2/BioLaySumm/checkpoint-best"
# config = PeftConfig.from_pretrained(peft_model_id)

# # load base LLM model and tokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path).to(device) 
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# # Load the Lora model
# model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
# model.eval()


## worked LongT5
# tokenizer = AutoTokenizer.from_pretrained('Stancld/longt5-tglobal-large-16384-pubmed-3k_steps')
# model = AutoModelForSeq2SeqLM.from_pretrained('Stancld/longt5-tglobal-large-16384-pubmed-3k_steps').to(device)

# special_tokens_dict = {'eos_token': '</s>'}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# print('We have added', num_added_toks, 'tokens')
# model.resize_token_embeddings(len(tokenizer))
# assert tokenizer.eos_token == '</s>'
# model = PeftModel.from_pretrained(model, 
#                                        '/jet/home/zyou2/BioLaySumm/checkpoint-best', 
#                                        torch_dtype=torch.bfloat16,
#                                        is_trainable=False)

# model = PeftModelForSeq2SeqLM.from_pretrained(
#     model,  # The base model to be used for prompt tuning
#     "/jet/home/zyou2/BioLaySumm/checkpoint-best",   # The path where the trained Peft model is saved
#     # torch_dtype=torch.bfloat16,
#     is_trainable=False  # Indicates that the loaded model should not be trainable
# ).to(device)
# model = model.merge_and_unload()

## worked
# peft_config = PeftConfig.from_pretrained('/jet/home/zyou2/BioLaySumm/checkpoint-best')
# peft_config.init_lora_weights = False

# model.add_adapter(peft_config)
# model.enable_adapters()

# model.load_adapter('/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/16384/longt5_elife/checkpoint-best/', model_name='Stancld/longt5-tglobal-large-16384-pubmed-3k_steps')


# 'google/long-t5-tglobal-base'
# model = AutoModelForSeq2SeqLM.from_pretrained("/jet/home/zyou2/BioLaySumm/LongT5-fine-tuning/16384_base/longt5_elife/checkpoint-1000/").to(device)
# model = LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps").to(device).half()
# AutoModelForSeq2SeqLM


# LED
tokenizer = AutoTokenizer.from_pretrained("/ocean/projects/cis230089p/zyou2/BioNLP/4k_led_base_extract_not_clean_0401/checkpoint-250/")
model = LEDForConditionalGeneration.from_pretrained("/ocean/projects/cis230089p/zyou2/BioNLP/4k_led_base_extract_not_clean_0401/checkpoint-250/").to(device)

def generate_sum(batch):
    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=4096, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)
    global_attention_mask = torch.zeros_like(attention_mask)
    # # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    # inputs_dict = tokenizer(
    #     batch["article"], max_length=16384, padding="max_length", truncation=True, return_tensors="pt"
    # ).to("cuda:0")
    
    # inputs_dict = tokenizer(
    #     batch["article"], max_length=16384, padding=True, truncation=True, return_tensors="pt"
    # ).to("cuda:0")
    
    # input_ids = inputs_dict.input_ids.to("cuda:0")
    # attention_mask = inputs_dict.attention_mask.to("cuda:0")
    # predicted_abstract_ids = model.generate(**inputs_dict, generation_config=GenerationConfig(max_new_tokens=512, num_beams=2))
    # predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=2)
    # predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id) #.sequences # eos_token_id=tokenizer.eos_token_id
    # predicted_abstract_ids = model.generate(**inputs_dict, max_new_tokens=512, num_beams=1)
    
    # LED
    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batch


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

### PLOS
# train
# plos_article_train, plos_lay_sum_train, plos_keyword_train, plos_headings_train, plos_id_train = load_data('PLOS', 'train')
# # val
# plos_article_val, plos_lay_sum_val, plos_keyword_val, plos_headings_val, plos_id_val = load_data('PLOS', 'val')
# # test
# plos_article_test, plos_keyword_test, plos_headings_test, plos_id_test = load_test_data('PLOS', 'test')


### eLife
# train
# elife_article_train, elife_lay_sum_train, elife_keyword_train, elife_headings_train, elife_id_train = load_data('eLife', 'train')
# val
elife_article_val, elife_lay_sum_val, elife_keyword_val, elife_headings_val, elife_id_val = load_data('eLife', 'val')
# test
# elife_article_test, elife_keyword_test, elife_headings_test, elife_id_test = load_test_data('eLife', 'test')


### using extractive sum
new_elife_article_val = []
new_elife_lay_sum_val = []

with open('/ocean/projects/cis230089p/zyou2/BioNLP/extract_aug_data/elife_val_new.json', 'r') as file:
    new_elife_article_val = json.load(file)

sub_elife_article_val = []
for text in new_elife_article_val:
    # result = re.sub(pattern, '', text)
    result = text.replace(' , ', ', ')
    sub_elife_article_val.append(result)
# val
elife_val_dataset = {'article': sub_elife_article_val, 'abstract': elife_lay_sum_val}
elife_val_dataset = Dataset.from_dict(elife_val_dataset)


### 8k input
# section_order = {
#     'abstract': ['abstract'],
#     'background': background,
#     'conclusions': conclusions,
#     'results': results,
#     'methods': methods
# }

# new_elife_article_val = []
# new_elife_lay_sum_val = []
# for lay, article, headings in zip(elife_lay_sum_val, elife_article_val, elife_headings_val):
#     sections = article.split('\n')
#     temp_sections = []
#     sections_dict = {section: '' for section in section_order}
#     for heading, section in zip(headings, sections):
#         heading = heading.lower()
#         for section_name, potential_headings in section_order.items():
#             if any(potential_heading in heading for potential_heading in potential_headings):
#                 sections_dict[section_name] += section + '\n'  # Append section content to the corresponding section in the dictionary
#                 break
#     for section_name in section_order:
#         temp_sections.append(sections_dict[section_name])
    
#     final_string = '\n'.join(temp_sections)
#     if final_string:
#         new_elife_article_val.append(final_string)
#         new_elife_lay_sum_val.append(lay)

# elife_val_dataset = {'article': new_elife_article_val, 'abstract': elife_lay_sum_val}
# elife_val_dataset = Dataset.from_dict(elife_val_dataset)
### end


# for lay, article, headings in zip(elife_lay_sum_val, elife_article_val, elife_headings_val):
#     sections = article.split('\n')
#     temp_sections = []
#     for i, (heading, section) in enumerate(zip(headings, sections)):
#         heading = heading.lower()
#         if heading in background:
#             temp_sections.append(section)
#         elif heading in conclusions:
#             temp_sections.append(section)
#         elif heading in 'abstract':
#             temp_sections.append(section)
#         else:
#             continue
#     final_string = '\n'.join(temp_sections)
#     if final_string:
#         new_elife_article_val.append(final_string)
#         new_elife_lay_sum_val.append(lay)
        

# plos val extract 
# new_elife_article_val = []
# with open('/ocean/projects/cis230089p/zyou2/BioNLP/extract_aug_data/plos_val.json', 'r') as file:
#     new_elife_article_val = json.load(file)

# print(len(new_elife_article_val))
# plos_val_dataset = {'article': new_elife_article_val, 'abstract': plos_lay_sum_val}
# plos_val_dataset = Dataset.from_dict(plos_val_dataset)



print("length of train and val datasets")
print(len(elife_val_dataset))


result = elife_val_dataset.map(generate_sum, batched=True, batch_size=16)
print(result["predicted_abstract"][2])

predicted_abstract = result["predicted_abstract"]

with open('/jet/home/zyou2/BioLaySumm/elife_4k_led_base_no_clean_extract.txt', 'w') as file:
    for abstract in predicted_abstract:
        file.write(abstract + '\n')
        
print("finished writing")
