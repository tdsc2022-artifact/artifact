'''cfg generator

@author : jumormt
@version : 1.0
'''
__author__ = "jumormt"
from utils.svf_parser import buildCFGNodeToLineDict, getCFGsEntryLineToNode
from networkx.drawing.nx_pydot import read_dot
import os
import json
import xml.etree.ElementTree as ET
from os.path import join
from typing import List, Set
from utils.unique import unique_cfg_list
from utils.clean_gadget import clean_gadget
from utils.vectorize_gadget import GadgetVectorizer
from sklearn.model_selection import train_test_split
from utils.git_checkout import checkout_to, checkout_back


def getCodeIDtoPathDict(testcases, sourceDir):
    '''build code testcaseid to path map

    use the manifest.xml. build {testcaseid:{filePath:set(vul lines)}}
    filePath use relevant path, e.g., CWE119/cve/source-code/project_commit/...
    :param testcases:
    :return: {testcaseid:{filePath:set(vul lines)}}
    '''
    codeIDtoPath = dict()
    for testcase in testcases:
        files = testcase.findall("file")
        testcaseid = testcase.attrib["id"]
        codeIDtoPath[testcaseid] = dict()

        for file in files:
            path = file.attrib["path"]
            flaws = file.findall("flaw")
            mixeds = file.findall("mixed")
            fix = file.findall("fix")
            # print(mixeds)
            VulLine = set()
            if (flaws != [] or mixeds != [] or fix != []):
                # targetFilePath = path
                if (flaws != []):
                    for flaw in flaws:
                        VulLine.add(int(flaw.attrib["line"]))
                if (mixeds != []):
                    for mixed in mixeds:
                        VulLine.add(int(mixed.attrib["line"]))

            codeIDtoPath[testcaseid][path] = VulLine

    return codeIDtoPath


def extract_cfgs(cfg_path: str, file_path_list: List[str]):
    '''

    :param: cfg_path:
    :param: file_path_list:
    :return: [{"nodes":[["1_1","1,2"], ["2_1","2_2"]], "edges":}]
    '''
    cfgs = list()
    ICFG = read_dot(cfg_path)
    CFGsNodeToLineDict = buildCFGNodeToLineDict(
        ICFG, file_path_list)  # for cfg:{nodeid:line}
    entryDict = getCFGsEntryLineToNode(
        ICFG, file_path_list)  # for cfg entry nodes:{line:nodeid}

    for key in entryDict:  # add entry nodeid to line info
        CFGsNodeToLineDict[entryDict[key]] = key

    for entry_node_id in entryDict.values():
        cfg = merge_cfg(entry_node_id, ICFG, CFGsNodeToLineDict)
        if (len(cfg["nodes"]) == 0):
            print("empty function!")
            continue
        cfgs.append(cfg)

    return cfgs


class BB:
    '''
    basic block
    '''
    def __init__(self):
        self.pred: Set[BB] = set()
        self.succ: Set[BB] = set()
        self.lines: Set[str] = set()


def dfs_merge_cfg(cur_node_id, ICFG, CFGsNodeToLineDict, cur_bb, visited,
                  bb_list):

    succ_nodes = ICFG._succ[cur_node_id]

    for succ_node_id in succ_nodes:
        if succ_node_id not in visited:
            if (len(succ_nodes) > 1 or len(ICFG._pred[succ_node_id]) > 1):
                n_bb = BB()
                bb_list.append(n_bb)
                visited[succ_node_id] = n_bb
                if succ_node_id in CFGsNodeToLineDict:
                    n_bb.lines.add(CFGsNodeToLineDict[succ_node_id])
                cur_bb.succ.add(n_bb)
                n_bb.pred.add(cur_bb)
                dfs_merge_cfg(succ_node_id, ICFG, CFGsNodeToLineDict, n_bb,
                              visited, bb_list)
            else:
                visited[succ_node_id] = cur_bb
                if succ_node_id in CFGsNodeToLineDict:
                    cur_bb.lines.add(CFGsNodeToLineDict[succ_node_id])
                dfs_merge_cfg(succ_node_id, ICFG, CFGsNodeToLineDict, cur_bb,
                              visited, bb_list)
        else:
            cur_bb.succ.add(visited[succ_node_id])
            visited[succ_node_id].pred.add(cur_bb)


def merge_cfg(entry_node_id, ICFG, CFGsNodeToLineDict):
    bb_list = list()
    entry_bb = BB()
    bb_list.append(entry_bb)
    entry_bb.lines.add(CFGsNodeToLineDict[entry_node_id])
    visited = dict()
    visited[entry_node_id] = entry_bb
    dfs_merge_cfg(entry_node_id, ICFG, CFGsNodeToLineDict, entry_bb, visited,
                  bb_list)
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

    return cfg


def generate_CFG(config):
    '''
    TODO: open-source projects
    :param config:
    :return:
    '''
    print("[start program]")
    cwEid = config.dataset.name
    data_root_dir = config.raw_data_folder
    cwe_root_dir = join(data_root_dir, 'CWE')
    dot_root_dir = join(cwe_root_dir, cwEid, "sard", "dot")
    if ((not os.path.exists(dot_root_dir)) or len(os.listdir(dot_root_dir)) == 0):
        print("no dot file!")
        return
    source_root_dir = join(cwe_root_dir, cwEid, "sard", "source-code")
    # outputDir = rootDir + "json-raw3/"
    data_out_dir = join(config.data_folder, config.name, cwEid)
    data_path = join(data_out_dir, "all.json")

    if (not os.path.exists(data_out_dir)):
        os.system(f"mkdir -p {data_out_dir}")
    xmlPath = join(source_root_dir, "manifest.xml")
    tree = ET.ElementTree(file=xmlPath)
    testcases = tree.findall("testcase")
    print("start - generating codeIDtoPath dict...")
    codeIDtoPath = getCodeIDtoPathDict(
        testcases, source_root_dir)  # {testcaseid:{filePath:set(vul lines)}}
    print("end - generating codeIDtoPath dict...")

    # =========================================================#
    # walk the testcase dir(subdirs : [testcaseid/dot files])  #
    # =========================================================#
    cfg_list = list()
    for root, testcaseIDList, files in os.walk(dot_root_dir):
        for testcaseID in testcaseIDList:
            file_path_list = list(codeIDtoPath[testcaseID].keys())

            dotRoot = join(root, testcaseID)
            cfg_path = join(dotRoot, "icfg_final.dot")
            print("start - processing files {} id {}...".format(
                file_path_list, testcaseID))
            # [{"nodes":[["1_1","1_2"],["2_1", "2_2"]], "edges":, "target":}]
            cfgs = extract_cfgs(cfg_path, file_path_list)

            print("end - processing files {}...".format(file_path_list))

            file_contents = list()
            for file_path in file_path_list:
                with open(join(source_root_dir, file_path),
                          "r",
                          encoding="utf-8",
                          errors="ignore") as f:
                    file_contents.append(f.readlines())
            print("start - symbolize, tokenize and label cfg..")
            for cfg in cfgs:
                offset = 0
                slices = list()
                all_lines = list()
                target = 0
                for bb in cfg["nodes"]:
                    slices.append(slice(offset, offset + len(bb)))
                    offset += len(bb)
                    for line in bb:
                        lineSplit = line.split("_")
                        fileidx = int(lineSplit[0])
                        fileContent = file_contents[fileidx]
                        content = fileContent[int(lineSplit[1]) - 1].strip()
                        all_lines.append(content)
                        if target == 1:
                            continue
                        if (int(lineSplit[1]) in codeIDtoPath[testcaseID][
                                file_path_list[fileidx]]):
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
                cfg_list.append(cfg)
            print("done!")

    # remove duplicates and conflicts
    print("start - unique cfg..")
    md5Dict = unique_cfg_list(cfg_list)
    cfg_list_unique = list()
    for mdd5 in md5Dict:
        if (md5Dict[mdd5]["target"] != -1):  # dont write conflict sample
            cfg_list_unique.append(md5Dict[mdd5]["cfg"])
    print("done!")
    with open(data_path, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(cfg_list_unique, f, indent=2)

    # # split dataset and output json
    # print("start - split cfg...")
    # cfg_list_unique = cfg_list_unique
    # X_train, X_test = train_test_split(cfg_list_unique, test_size=0.2)
    # X_test, X_val = train_test_split(
    #     X_test,
    #     test_size=0.5,
    # )
    # print("done!")

    # with open(train_data_path, "w", encoding="utf-8", errors="ignore") as f:
    #     json.dump(X_train, f, indent=2)
    # with open(test_data_path, "w", encoding="utf-8", errors="ignore") as f:
    #     json.dump(X_test, f, indent=2)
    # with open(val_data_path, "w", encoding="utf-8", errors="ignore") as f:
    #     json.dump(X_val, f, indent=2)
    print("[end program]")
    return


def generate_CFG_osp(config):
    '''
    TODO: open-source projects
    :param config:
    :param project: redis
    :return:
    '''
    print("[start program]")
    project = config.dataset.name
    data_root_dir = config.raw_data_folder
    cve_root_dir = join(data_root_dir, 'CVE')
    dot_root_dir = join(cve_root_dir, project, "dot")
    source_root_dir = join(cve_root_dir, project, "source-code")
    repo_dir = join(source_root_dir, project)
    # outputDir = rootDir + "json-raw3/"
    data_out_dir = join(config.data_folder, config.name, project)
    data_path = join(data_out_dir, "all.json")

    if (not os.path.exists(data_out_dir)):
        os.system(f"mkdir -p {data_out_dir}")
    xmlPath = join(source_root_dir, "manifest.xml")
    tree = ET.ElementTree(file=xmlPath)
    testcases = tree.findall("testcase")
    print("start - generating codeIDtoPath dict...")
    codeIDtoPath = getCodeIDtoPathDict(
        testcases, source_root_dir)  # {testcaseid:{filePath:set(vul lines)}}
    print("end - generating codeIDtoPath dict...")

    # =========================================================#
    # walk the testcase dir(subdirs : [testcaseid/dot files])  #
    # =========================================================#
    cfg_list = list()
    for root, testcaseIDList, files in os.walk(dot_root_dir):
        for testcaseID in testcaseIDList:
            git_r = checkout_to(repo_dir, testcaseID, project)
            file_path_list = list(codeIDtoPath[testcaseID].keys())
            dotRoot = join(root, testcaseID)
            for dot_dir in os.listdir(dotRoot):
                cfg_path = join(dotRoot, dot_dir, "icfg_final.dot")
                print("start - processing files {} id {}...".format(
                file_path_list, testcaseID))
                # [{"nodes":[["1_1","1_2"],["2_1", "2_2"]], "edges":, "target":}]
                cfgs = extract_cfgs(cfg_path, file_path_list)

                print("end - processing files {}...".format(file_path_list))

                file_contents = list()
                for file_path in file_path_list:
                    with open(join(source_root_dir, project, file_path),
                        "r",
                        encoding="utf-8",
                        errors="ignore") as f:
                        file_contents.append(f.readlines())
                print("start - symbolize, tokenize and label cfg..")
                for cfg in cfgs:
                    offset = 0
                    slices = list()
                    all_lines = list()
                    target = 0
                    for bb in cfg["nodes"]:
                        slices.append(slice(offset, offset + len(bb)))
                        offset += len(bb)
                        for line in bb:
                            lineSplit = line.split("_")
                            fileidx = int(lineSplit[0])
                            fileContent = file_contents[fileidx]
                            content = fileContent[int(lineSplit[1]) - 1].strip()
                            all_lines.append(content)
                            if target == 1:
                                continue
                            if (int(lineSplit[1]) in codeIDtoPath[testcaseID][
                                    file_path_list[fileidx]]):
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
                    cfg_list.append(cfg)
            checkout_back(git_r, project)
            print("done!")

    # remove duplicates and conflicts
    print("start - unique cfg..")
    md5Dict = unique_cfg_list(cfg_list)
    cfg_list_unique = list()
    for mdd5 in md5Dict:
        if (md5Dict[mdd5]["target"] != -1):  # dont write conflict sample
            cfg_list_unique.append(md5Dict[mdd5]["cfg"])
    print("done!")
    with open(data_path, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(cfg_list_unique, f, indent=2)


    # # split dataset and output json
    # print("start - split cfg...")
    # cfg_list_unique = cfg_list_unique
    # X_train, X_test = train_test_split(cfg_list_unique, test_size=0.2)
    # X_test, X_val = train_test_split(
    #     X_test,
    #     test_size=0.5,
    # )
    # print("done!")

    # with open(train_data_path, "w", encoding="utf-8", errors="ignore") as f:
    #     json.dump(X_train, f, indent=2)
    # with open(test_data_path, "w", encoding="utf-8", errors="ignore") as f:
    #     json.dump(X_test, f, indent=2)
    # with open(val_data_path, "w", encoding="utf-8", errors="ignore") as f:
    #     json.dump(X_val, f, indent=2)
    print("[end program]")
    return
