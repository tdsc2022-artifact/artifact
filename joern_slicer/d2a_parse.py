import csv
import os
from utils.json_ops import write_json, read_json
from utils.common import get_config
import shutil
from numpy import random
import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
import json
import numpy as np
from utils.git_checkout import checkout_to
import numpy
from joern_slicer.slicing import get_slice_for_cdg
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
projects = ['ffmpeg', 'httpd', 'libav', 'nginx', 'openssl']
from utils.vectorize_gadget import GadgetVectorizer
from utils.clean_gadget import clean_gadget
from utils.unique import getMD5
def d2a_classify_as_bug_type():
    """
    @description  :classsify d2a as bug type 
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    
    label_info = dict()
    rears = ['_labeler_1.json']
    d2a_src_dir = '/home/chengxiao/dataset/vul/d2a/d2a_json'
    d2a_label_json_path = '/home/chengxiao/dataset/vul/d2a/label_info.json'
    for project in projects:
        for rear in rears:
            if rear == '_labeler_1.json':
                info_type = 'vul'
            json_file_path = os.path.join(d2a_src_dir, project + rear) 
            # print(json_file_path)
            json_list = read_json(json_file_path)
            # print(json_list)
            for json_str in json_list:
                label = json_str['label']
                sample_type = json_str['sample_type']
                bug_type = json_str['bug_type'] 
                project = json_str['project']
                bug_info = json_str['bug_info']
                trace = json_str['trace']
                functions = json_str['functions']
                adjusted_bug_loc = json_str['adjusted_bug_loc']
                if sample_type == 'before_fix':
                    commit_id = json_str['versions']['before']
                elif sample_type == 'after_fix':
                    commit_id = json_str['versions']['after']

                if adjusted_bug_loc != None :
                    file_path = adjusted_bug_loc['file']
                    line = adjusted_bug_loc['line']
                else :
                    file_path = bug_info['file']
                    line = bug_info['line']
                key = project + '/' + commit_id
                if key not in label_info.keys():
                    label_info[key] = list()
                    
                info = dict()
                trace_lines = []
                for t in trace:
                    t_file = t['file']
                    t_loc = t['loc'].split(':')[0]
                    trace_lines.append(f'{t_file}_{t_loc}')
                info['file_path'] = file_path
                info['line'] = line
                info['trace_lines'] = trace_lines
                info['label'] = label
                label_info[key].append(info)
    write_json(label_info, d2a_label_json_path)
    
    # extract_label_file_code(label_json=label_info)

def extract_label_file_code(label_json):
    last_commit_id = ''
    for key in label_json:
        project = key.split('/')[0]
        commit_id = key.split('/')[1]
        for info in label_json[key]:
       
            file_path = info['file_path']
            source_path = os.path.join('/home/chengxiao/dataset/vul/d2a/source_code', project)
            target_path = os.path.join('/home/chengxiao/dataset/vul/d2a/tmp', project)
        

            source_file = os.path.join(target_path, file_path)
            target_file = os.path.join('/home/chengxiao/dataset/vul/d2a/extracted_code_and_slice/code/d2a_resources', project, commit_id, file_path)
            if os.path.exists(target_file):
                print(target_file + ' has done!')
                continue
            if last_commit_id != commit_id:
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                    print(target_path + ' exist ! remove it')
                shutil.copytree(source_path, target_path)
                checkout_to(target_path, commit_id, project)
            
            path = os.path.split(target_file)
            if not os.path.exists(path[0]):
                os.makedirs(path[0])
            shutil.copyfile(source_file, target_file)
            last_commit_id = commit_id

def get_cdg_d2a(info_dict, data_instance, method):
    info_list = []
    if not data_instance:
        return info_list
    file_path = data_instance['file_path']
    vul_lines = info_dict['line']
    trace_lines = info_dict['trace_lines']
    if method == 'sysevr':
        slices = data_instance['all_slices_sy']
    elif method == 'vuldeepecker':
        slices = data_instance['all_slices_vd']
    file_content = data_instance['file_content']

    
    for slis in slices:
        if(len(slis) < 3):
            continue
        info = dict()
        vul_line = 0
        flag = False
        for line in slis:
            if int(line) == vul_lines:
                flag = True
                vul_line = int(line)
                break
        if flag:
            target = 1
        else:
            target = 0
            # info = file_path + '|' + 'vul_line:{}'.format(line) + '|' + 'pair_id:{}\n'.format(pair_id)
            # content = get_slice_content(file_content, slis)                   
        # if type == 'vul':
        #     target = 1
        # else:
        #     target = 0
              
        if target == 1:
            sd_line_set = set(['{}_{}'.format(info_dict['file_path'], s) for s in slis])
            sp_line_set = set(trace_lines) 
            info['sp&sd'] = len(sd_line_set & sp_line_set) 
            info['sp|sd'] = len(sd_line_set | sp_line_set) 
            info['sp'] = len(sp_line_set)
            info['sd'] = len(slis)
        else:
            info['sp&sd'] = 0
            info['sp|sd'] = len(slis)
            info['sp'] = 0
            info['sd'] = len(slis)    
        info['file_path'] = file_path
        info['vul_line'] = vul_line
        info['content'] = get_slice_content(file_content, slis)
        info['target'] = target
        info_list.append(info)
            # info_list.append(info + content + str(target) + "\n" +
            #                             "---------------------------------" + "\n")

    return info_list       

def get_slice_content(file_content, slis):
    """
    @description  : get file statements for slice
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    content = []
    for line in slis:
        # content.append(file_content[line-1].strip())
        content.append(file_content[line-1].strip())  
    return content



def sys_cdg_duplicate(config, holdout_data_path):
    if (not os.path.exists(holdout_data_path)):
        print(f"there is no file named: {holdout_data_path}")
        return
    sensi_api_path = 'joern_slicer/resources/sensiAPI.txt'
    vocab_path = os.path.join(config.data_folder, config.name,
                           config.dataset.name, "vocab.pkl")   
    data_path = os.path.join(config.data_folder, config.name,
                           config.dataset.name, "{}.json".format(config.dataset.name))   
    if (not os.path.exists(os.path.dirname(data_path))):
        os.makedirs(os.path.dirname(data_path))
    gadgets = dict()  # {md5:}
    count = 0
    dulCount = 0
    mulCount = 0
    vectorizer = GadgetVectorizer(config)
    cdg_list = read_json(holdout_data_path)
    for cdg in cdg_list:
        gadget = clean_gadget(cdg['content'])
        val = cdg['target']
        count += 1
        print("Collecting gadgets...", count, end="\r")
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(
            gadget)
        tokenized_gadget_md5 = getMD5(str(tokenized_gadget))
        if (tokenized_gadget_md5 not in gadgets):
            row = {"gadget": gadget, 
                   "val": val, 
                   "count": 0, 
                   "file_path":cdg['file_path'], 
                   "vul_line":cdg['vul_line'],
                   "sp":cdg['sp'],
                   "sd":cdg['sd'],
                   "sp&sd":cdg['sp&sd'],
                   "sp|sd":cdg['sp|sd']
                   }
            gadgets[tokenized_gadget_md5] = row
        else:
            if (gadgets[tokenized_gadget_md5]["val"] != -1):
                if (gadgets[tokenized_gadget_md5]["val"] != val):
                    dulCount += 1
                    gadgets[tokenized_gadget_md5]["val"] = -1
            mulCount += 1
        gadgets[tokenized_gadget_md5]["count"] += 1
    print('\n')
    print("Find multiple...", mulCount)
    print("Find dulplicate...", dulCount)
    gadgets_unique = list()
    for gadget_md5 in gadgets:
        if (gadgets[gadget_md5]["val"] != -1):  # remove dulplicated
            vectorizer.add_gadget(gadgets[gadget_md5]["gadget"])
            gadgets_unique.append(gadgets[gadget_md5])
            # for i in range(gadgets[gadget_md5]["count"]):# do not remove mul
            #     vectorizer.add_gadget(gadgets[gadget_md5]["gadget"])
    # print('Found {} forward slices and {} backward slices'.format(
    #     vectorizer.forward_slices, vectorizer.backward_slices))

    print("Training word2vec model...", end="\r")
    w2v_path = os.path.join(config.data_folder, config.name, config.dataset.name,
                         "w2v.model")
    
    vectorizer.train_model(w2v_path)
    vectorizer.build_vocab(vocab_path)    
    
    gadget_vul = list()
    gadget_safe = list()
    for gadget in gadgets_unique:
        if gadget['val'] == 1:
            gadget_vul.append(gadget)
        else:
            gadget_safe.append(gadget)
    # numpy.random.seed(7)
    # if len(gadget_safe) >= len(gadget_vul) * 2:
    #     sub_safe = numpy.random.choice(gadget_safe, len(gadget_vul)*2, replace=False)
    # else:
    #     sub_safe = gadget_safe
    all_gadgets = []
    all_gadgets.extend(gadget_vul)
    all_gadgets.extend(gadget_safe)
    numpy.random.shuffle(all_gadgets)
    print(len(gadget_vul), len(gadget_safe), len(all_gadgets))

    write_json(all_gadgets, data_path)
    
def joern_parse(data_path, cwe_id, gen_csv:bool=False):
    """
    @description  : use joern to parse c/cpp
    ---------
    @param  : data_path: c/cpp dir
    -------
    @Returns  : 
    output: joern/output
    os.path.abspath(data_path) : data_path absolute path 
    note: output + os.path.abspath(data_path) is csv_path
    -------
    """
    output = CUR_DIR + '/joern/output_{}'.format(cwe_id)
    
    cmd = CUR_DIR + '/joern/joern-parse {} {}'.format(output, data_path)
    print('CMD: '+cmd)
    if gen_csv:
        os.system(cmd)
    return output, os.path.abspath(data_path)



def d2a_cdg_label(d2a_bug_type, method, gen_csv:bool=False):
    source_code_dir = '/home/chengxiao/dataset/vul/d2a/extracted_code_and_slice/code/d2a_resources'
    label_list_path = '/home/chengxiao/dataset/vul/d2a/label_info.json'
    label_info_dict = read_json(label_list_path)
    
    output_dir, abs_data_path = joern_parse(source_code_dir, d2a_bug_type, gen_csv)

    # output_dir = CUR_DIR + '/joern/output_{}'.format('d2a')

    all_info_list = []

    for key in label_info_dict:
        
        for info in label_info_dict[key]:
            vul_info = dict()
            vul_info['path'] = os.path.join(key, info['file_path'])
            vul_info['line'] = info['line']
            data_instance = get_slice_for_cdg(vul_info, output_dir, abs_data_path)

            if data_instance == None:
                continue      
            info_list = get_cdg_d2a(info, data_instance, method)
            if info_list != [] :
                all_info_list.extend(info_list)
    return all_info_list

def gen_cdg_raw_data_d2a(cwe_id, method, is_gen_csv=False):
      
    config = get_config('sysevr', 'd2a')

    raw_data_path = os.path.join(config.data_folder, config.name, config.dataset.name, 'raw.json')
    # generate raw data

    gadgets = d2a_cdg_label(cwe_id, config.name, is_gen_csv)
    write_json(gadgets, raw_data_path)
    print("end")
    # unique
    print('unique')

    sys_cdg_duplicate(config, holdout_data_path=raw_data_path)