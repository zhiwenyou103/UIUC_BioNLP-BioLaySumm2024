import os
import json
import argparse 
from tqdm import tqdm
import textstat

def get_data(file_path):
    f = open(file_path)
    data = []
    for line in f.readlines():
        # sentences division and clean
        data.append(line.strip())
    f.close()
    
    return data

def load_task1_data(args):

    data_folder = './task1_development'
    data_path = os.path.join(data_folder, args.datatype)
    data_path = os.path.join(data_path, f'{args.dataset}_{args.datatype}.jsonl')

    lay_sum = []
    article =[]

    keyword = []

    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        article.append(dic['article'])
        lay_sum.append(dic['lay_summary'])
        keyword.append(dic['keywords'])
    
    return article, lay_sum, keyword

def load_task2_data():
    data_folder = './task2_development'
    datatype = 'val'
    data_path = os.path.join(data_folder, f'{datatype}.jsonl')
    
    abstract = []
    lay_sum = []
    
    file = open(data_path, 'r')
    for line in (file.readlines()):
        dic = json.loads(line)
        abstract.append(dic['abstract'])
        lay_sum.append(dic['lay_summary'])
    
    return abstract, lay_sum

def check_keyword(lay_sum, keyword):
    ans = len(keyword)
    tmp = 0
    for word in keyword:
        if(word in lay_sum):
            tmp += 1
    if(tmp==0):
        return True
    else:
        return False
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='PLOS')
    parser.add_argument("--datatype", type=str, default="train")
    args = parser.parse_args()

    article, lay_sum, keyword = load_task1_data(args)
    
    result = 0
    result_1 = 0
    for i in range(len(lay_sum)):
        if(check_keyword(lay_sum[i], keyword[i])):
            result += 1
        else:
            result_1 += 1
    
    print(result, result_1)
        
