import os
from os.path import join, isdir
import csv
import networkx as nx
from typing import List, Set, Dict
import sys
sys.path.append("..")
from utils.xml_parser import getCodeIDtoPathDict_osp, getCodeIDtoPathDict_d2a
from utils.clean_gadget import clean_gadget
from utils.vectorize_gadget import GadgetVectorizer
from utils.unique import unique_cfg_list
import xml.etree.ElementTree as ET
import json
import copy
import jsonlines
from utils.json_ops import read_json, write_json
from joern_slicer.d2a_parse import joern_parse
from utils.common import get_config
def extract_line_number(idx, nodes):
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    pass
        idx -= 1
    return -1


def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def extract_nodes_with_location_info(nodes):
    # Will return an array identifying the indices of those nodes in nodes array,
    # another array identifying the node_id of those nodes
    # another array indicating the line numbers
    # all 3 return arrays should have same length indicating 1-to-1 matching.
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    functions = set()  # function line number
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            if node['type'] == "Function":
                functions.add(line_num)
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number, functions


class BB:
    '''
    basic block
    '''
    def __init__(self):
        self.pred: Set[BB] = set()
        self.succ: Set[BB] = set()
        self.lines: Set[str] = set()


def dfs_merge_cfg(cur_node_ln, ICFG, cur_bb, visited, bb_list):

    succ_nodes = ICFG._succ[cur_node_ln]

    for succ_node_ln in succ_nodes:
        if succ_node_ln not in visited:
            if (len(succ_nodes) > 1 or len(ICFG._pred[succ_node_ln]) > 1):
                n_bb = BB()
                bb_list.append(n_bb)
                visited[succ_node_ln] = n_bb
                n_bb.lines.add(succ_node_ln)
                cur_bb.succ.add(n_bb)
                n_bb.pred.add(cur_bb)
                dfs_merge_cfg(succ_node_ln, ICFG, n_bb, visited, bb_list)
            else:
                visited[succ_node_ln] = cur_bb

                cur_bb.lines.add(succ_node_ln)
                dfs_merge_cfg(succ_node_ln, ICFG, cur_bb, visited, bb_list)
        else:
            cur_bb.succ.add(visited[succ_node_ln])
            visited[succ_node_ln].pred.add(cur_bb)


def extract_cfgs(file_path: str) -> List[Dict[str, List[List[int]]]]:
    r"""
    extract cfgs of the file specified by file_path

    Args:
        file_path (str): file path to be extracted

    Return:
        cfgs (List[Dict[str, List[List[int]]]])

    Examples:
        .. code-block:: python
            cfgs = extract_cfgs("test_cfg.cpp")

    Note:
        XXX
    """

    nodes_path = join(file_path, "nodes.csv")
    edges_path = join(file_path, "edges.csv")
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    node_indices, node_ids, line_numbers, node_id_to_ln, functions = extract_nodes_with_location_info(
        nodes)
    ICFG_edges = set()
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_ln.keys(
            ) or end_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = node_id_to_ln[end_node_id]

            if edge_type == 'FLOWS_TO':  # Control Flow edges
                ICFG_edges.add((start_ln, end_ln))

    ICFG = nx.DiGraph()
    ICFG.add_edges_from(ICFG_edges)
    cfgs = list()
    # for each cfg entry
    functions = sorted(functions)
    if len(ICFG._node.keys()) == 0:
        return cfgs
    max_ln = max(ICFG._node.keys())
    for idx, entry_ln in enumerate(functions):
        
        is_empty = False
        while entry_ln not in ICFG._node:
            if (idx + 1 < len(functions) and entry_ln >= functions[idx + 1]
                ) or entry_ln > max_ln:
                is_empty = True
                break
            entry_ln += 1

        if (is_empty):
            continue
        bb_list = list()

        entry_bb = BB()
        bb_list.append(entry_bb)
        entry_bb.lines.add(entry_ln)
        visited = dict()
        visited[entry_ln] = entry_bb
        dfs_merge_cfg(entry_ln, ICFG, entry_bb, visited, bb_list)
        bb_to_idx = dict()
        cfg = dict()
        cfg["nodes"] = list()
        cfg["edges"] = list()
        # remove empty basic block
        bb_list_cp = bb_list.copy()
        for bb in bb_list_cp:
            if len(bb.lines) == 0:
                bb_list.remove(bb)
        for pos, bb in enumerate(bb_list):
            bb_to_idx[bb] = pos
            cfg["nodes"].append(list(bb.lines))
        for pos, bb in enumerate(bb_list):
            for succ_bb in bb.succ:
                if succ_bb in bb_to_idx:
                    cfg["edges"].append([pos, bb_to_idx[succ_bb]])
        cfgs.append(cfg)
    return cfgs

def d2a_cfg_label(dataset, method, gen_csv:bool=False):
    source_code_dir = '/home/chengxiao/dataset/vul/d2a/extracted_code_and_slice/code/d2a_resources'
    label_list_path = '/home/chengxiao/dataset/vul/d2a/label_info.json'
    label_info_dict = read_json(label_list_path)
    output_dir, abs_data_path = joern_parse(source_code_dir, dataset, gen_csv)
    all_cfgs = []
    for key in label_info_dict:
        for info in label_info_dict[key]:
            file_path = os.path.join(key, info['file_path'])
            src = os.path.join(abs_data_path, file_path)
            csv_root = output_dir + src
            cfgs = extract_cfgs(csv_root)
            vul_line = info['line']
            
            trace_lines = info['trace_lines']
            with open(src,
                                      "r",
                                      encoding="utf-8",
                                      errors="ignore") as f:
                                fileContent = f.readlines()
            
            for cfg in cfgs:
                offset = 0
                slices = list()
                all_lines = list()
                all_line_number = list()
                target = 0  # not correct
                for bb in cfg["nodes"]:
                    slices.append(slice(offset, offset + len(bb)))
                    offset += len(bb)
                    for line in bb:
                        content = fileContent[int(line) - 1].strip()
                        all_lines.append(content)
                        all_line_number.append('{}_{}'.format(info['file_path'], line))
                        if target == 1:
                            continue
                        if int(line) == vul_line:
                            target = 1
                # symbolize cfg
                all_lines_sym = clean_gadget(all_lines)
                for sidx in range(len(slices)):
                    s = slices[sidx]
                    bb = all_lines_sym[s]
                    tokens = list()
                    for line in bb:
                        # tokenize cfg
                        tokens.extend(GadgetVectorizer.tokenize(line))
                    cfg["nodes"][sidx] = tokens
                cfg["target"] = target
                cfg["file_path"] = file_path
                if target == 1:
                    sd_line_set = set(all_line_number)
                    sp_line_set = set(trace_lines) 
                    cfg['sp&sd'] = len(sd_line_set & sp_line_set) 
                    cfg['sp|sd'] = len(sd_line_set | sp_line_set) 
                    cfg['sp'] = len(sp_line_set)
                    cfg['sd'] = len(all_line_number)
                else:
                    cfg['sp&sd'] = 0
                    cfg['sp|sd'] = len(all_line_number)
                    cfg['sp'] = 0
                    cfg['sd'] = len(all_line_number)
                
                if len(all_line_number) < 3:
                    continue
                all_cfgs.append(cfg)
    return all_cfgs
    
def generate_cfg_d2a():
    config = get_config('vgdetector', 'd2a')
    raw_data_path = os.path.join(config.data_folder, config.name, config.dataset.name, 'raw.json')
    cfg_out_path = os.path.join(config.data_folder, config.name, config.dataset.name, 'all.json')
    
    cfgs = d2a_cfg_label(config.dataset.name, config.name, False)
    write_json(cfgs, output=raw_data_path, is_need_create_dir=True)
    print("start - unique cfg..")
    md5Dict = unique_cfg_list(cfgs)
    cfg_list_unique = list()
    for mdd5 in md5Dict:
        if (md5Dict[mdd5]["target"] != -1):  # dont write conflict sample
            cfg_list_unique.append(md5Dict[mdd5]["cfg"])
    write_json(cfg_list_unique, cfg_out_path)
    print("done!")
# def generate_cfg_d2a(source_root_dir,project, xmlPath, cfg_out_path, resources_dir):
#     """
#     @description  :Extract the cfg of D2A's source code
#                    according to the organization form of XML, 
#                    and output it as a jsonline format file
#     --------
#     @param  :source_root_dir:source code dir;
#              project:project_name
#              xmlPath: xml file path;
#              cfg_out_path: cfg_out_dir
#              resources_dir:c/cpp source code dir
#     -------
#     @Returns  :
#     -------
#     """
    
    
#     tree = ET.ElementTree(file=xmlPath)
#     testcases = tree.findall("testcase")
#     print("start - generating codeIDtoPath dict...")
#     codeIDtoPath = getCodeIDtoPathDict_d2a(
#         testcases, source_root_dir)  # {testcaseid:{filePath:set(vul lines)}}
#     print(codeIDtoPath)
#     if not os.path.exists(cfg_out_path):
#         os.makedirs(cfg_out_path)
#     count = 0
#     for testcase in testcases:
#         testcaseID = testcase.attrib["id"]
#         filePathList = list(codeIDtoPath[testcaseID].keys())
#         #safe
#         testcaseDir = os.path.join(
#             resources_dir, project, testcaseID)
#         if not os.path.exists(testcaseDir):
#             continue
#         for file_dir in filePathList:
#             file_path = os.path.join(testcaseDir, file_dir)
#             with open(cfg_out_path + '/hasdone.txt', 'r') as f:
#                 hasdones = set(f.read().split("\n"))
#                 f.close()
#             if file_path in hasdones:
#                 print(file_path+ ' hasdone continue...')
#                 continue
#             if os.path.exists(file_path) and os.path.isdir(file_path):
#                 source_code_path = '/' + '/'.join(file_path.split('/')[3:])
#                 cfgs = extract_cfgs(file_path)
#                 cfgs_copy = copy.deepcopy(cfgs)
#                 key = file_dir
#                 with open(source_code_path,
#                                       "r",
#                                       encoding="utf-8",
#                                       errors="ignore") as f:
#                                 fileContent = f.readlines()
#                                 f.close()
#                 for cfg in cfgs:
#                     offset = 0
#                     slices = list()
#                     all_lines = list()
#                     target = -1  # not correct
#                     for bb in cfg["nodes"]:
#                         slices.append(slice(offset, offset + len(bb)))
#                         offset += len(bb)
#                         for line in bb:
#                             content = fileContent[int(line) - 1].strip()
#                             all_lines.append(content)
#                             if target == 0:
#                                 continue
#                             if str(line) in codeIDtoPath[testcaseID][key]['SafeLine'].keys():
#                                 target = 0
#                                 key_line = str(line)
#                     # symbolize cfg
#                     if target != 0:
#                         continue
#                     all_lines_sym = clean_gadget(all_lines)

#                     for sidx in range(len(slices)):
#                         s = slices[sidx]
#                         bb = all_lines_sym[s]
#                         tokens = list()
#                         for line in bb:
#                             # tokenize cfg
#                             tokens.extend(GadgetVectorizer.tokenize(line))
#                         cfg["nodes"][sidx] = tokens
#                     cfg["target"] = target
#                     cfg["file_path"] = file_path

#                     for bug_type in codeIDtoPath[testcaseID][key]['SafeLine'][key_line]:
#                             cfg_out_dir = os.path.join(cfg_out_path, bug_type)
#                             if not os.path.exists(cfg_out_dir):
#                                 os.mkdir(cfg_out_dir)
#                             count += 1
#                             f=jsonlines.open(cfg_out_dir + '/all.jsonl',"a")
#                             jsonlines.Writer.write(f,cfg) # 每行写入一个dict                            
#                             f.close()
#                             print(project+' writing cfg '+str(count))
#                             # print(str(cfg))
                    

                   
#                 #vul
#                 key = file_dir
#                 for cfg in cfgs_copy:
#                     offset = 0
#                     slices = list()
#                     all_lines = list()
#                     target = -1  # not correct
#                     for bb in cfg["nodes"]:
#                         slices.append(slice(offset, offset + len(bb)))
#                         offset += len(bb)
#                         for line in bb:
#                             content = fileContent[int(line) - 1].strip()
#                             all_lines.append(content)
#                             if target == 1:
#                                 continue
#                             if str(line) in codeIDtoPath[testcaseID][key]['VulLine'].keys():
#                                 key_line = str(line)
#                                 target = 1
#                     # symbolize cfg
#                     if target != 1:
#                         continue
#                     all_lines_sym = clean_gadget(all_lines)

#                     for sidx in range(len(slices)):
#                         s = slices[sidx]
#                         bb = all_lines_sym[s]
#                         tokens = list()
#                         for line in bb:
#                             # tokenize cfg
#                             tokens.extend(GadgetVectorizer.tokenize(line))
#                         cfg["nodes"][sidx] = tokens
#                     cfg["target"] = target
#                     cfg['file_path'] = file_path

#                     for bug_type in codeIDtoPath[testcaseID][key]['VulLine'][key_line]:
#                             cfg_out_dir = os.path.join(cfg_out_path, bug_type)
#                             if not os.path.exists(cfg_out_dir):
#                                 os.mkdir(cfg_out_dir)
#                             count += 1
#                             f=jsonlines.open(cfg_out_dir + '/all.jsonl',"a")
#                             jsonlines.Writer.write(f,cfg) # 每行写入一个dict                            
#                             f.close()
#                             print(project+' writing cfg '+str(count))

#             with open(cfg_out_path + '/hasdone.txt', 'a') as f:
#                 f.write(file_path+'\n')
#                 f.close()
        
                



#     return

def merge_cfg_from_jsonl(cfg_path):
    """
    @description  : merge cfg in jsonline format to json format, then de-duplication
    ---------
    @param  :cfg_path: the dir of cfg in jsonline format
    -------
    @Returns  :
    -------
    """
    
    
    #jsonl 转为 合为json 并且去重
    for bug_type in os.listdir(cfg_path):
        if os.path.isdir(os.path.join(cfg_path, bug_type)):
            jsonl = os.path.join(cfg_path, bug_type, 'all.jsonl')
            if os.path.exists(jsonl):
                # 读文件，不使用jsonlines.open() 而是直接open()
                f=open(jsonl,"r")
                reader=jsonlines.Reader(f)
                cfg_list = list()
                for cfg in reader:
                    cfg_list.append(cfg)
                # remove duplicates and conflicts
                print("start - unique cfg..")
                md5Dict = unique_cfg_list(cfg_list)
                cfg_list_unique = list()
                for mdd5 in md5Dict:
                    if (md5Dict[mdd5]["target"] != -1):  # dont write conflict sample
                        cfg_list_unique.append(md5Dict[mdd5]["cfg"])
                print("done!")
                cfg_out_path = os.path.join(cfg_path,bug_type)
                with open(cfg_out_path+'/all.json', "w", encoding="utf-8", errors="ignore") as f:
                    json.dump(cfg_list_unique, f, indent=2)
                    f.close()


if __name__ == "__main__":
    # sys.setrecursionlimit(10000)  # 将默认的递归深度修改为3000
    # root = 'joern/d2a_resources'
    # for project in os.listdir(root):
    #     # if project in  ['httpd','nginx']:
    #     #     continue
    #     try:
    #         generate_cfg_d2a(project)
    #     except Exception as e:
    #         with open('generate_cfg_d2a_error.log', 'a') as f:
    #             f.write(project+'\n')
    #             f.write(str(e)+'\n')
    #             f.close()
    # merge_cfg_from_jsonl()
    cfgs = extract_cfgs('output/main.cpp')
    print(cfgs)
    with open('main.json', 'w', encoding='utf8') as f:
        json.dump(cfgs, f, indent = 2)
        f.close()
        