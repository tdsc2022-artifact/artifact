import json
from os import makedirs
import os
import random
def write_json(json_dict, output, is_need_create_dir=False):
    print(f'Start writing json data in {output}')
    if is_need_create_dir:
        print(f'making dirs for {output}')
        makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf8') as f:
        json.dump(json_dict,f,indent=2)
        f.close()
    print(f'Writing successful !')
    

def read_json(path):
    print(f'Start reading json data from {path}')
    json_dict = []
    with open(path, 'r', encoding='utf8') as f:
        
        json_dict = json.load(f)
        f.close()
    print(f'Reading successful !')
    
    return json_dict

def get_data_json(data_path):
    random.seed(7)
    with open(data_path,'r',encoding = 'utf8') as f:
        data_json = json.load(f)
        f.close()
    random.shuffle(data_json)
    for idx,xfg in enumerate(data_json, start=0):
        xfg['xfg_id'] = idx
        xfg['flip'] = False
    return data_json

def write_file(data:list, path:str, is_need_create_dir:bool=False):
    if is_need_create_dir:
        print(f'making dirs for {path}')
        makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        print(f'writing data in to {path}')
        f.writelines(data)