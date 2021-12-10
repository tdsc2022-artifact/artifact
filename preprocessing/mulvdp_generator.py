'''mulvuldeepecker generator

backward and forward slicing control- and data-flow related statements
'''
from utils.svf_parser import buildCFGNodeToLineDict, getCFGsEntryLineToNode, buildSVFGNodeToLineDict, locateLineEntry, extractOperatorLines, buildCallGraphDict, extractAPILines, getCallChainList, dfs_pred_lines
from networkx.drawing.nx_pydot import read_dot
from flow_analysis.control.CFG_influence_builder import buildCFGDepedcyInflueceDict
import os
import json
import xml.etree.ElementTree as ET
from os.path import join
from utils.git_checkout import checkout_to, checkout_back
import networkx as nx
from typing import List, Dict, Set
from utils.xml_parser import getCodeIDtoPathDict


def buildCodeGadgetList(doneLines, svfgPath, cfgPath, sensiAPIPath,
                        filePathList, callgPath):
    '''

    :param doneLines:
    :param svfgPath:
    :param cfgPath:
    :param sensiAPIPath:
    :param filePathList:
    :param callgPath:
    :return:
    '''

    SVFG = read_dot(svfgPath)  # SVFG in networkx format
    CFGs = read_dot(cfgPath)  # CFGs in networkx format
    CallG = read_dot(callgPath)  # CallG in networkx format

    print("start - getting callgraph...")
    calleeToCallerDict = buildCallGraphDict(
        CFGs, CallG, filePathList)  # {callee line:set(caller line)}
    print("end - getting callgraph...")

    CFGsNodeToLineDict = buildCFGNodeToLineDict(
        CFGs, filePathList)  # for cfg:{nodeid:line}
    entryDict = getCFGsEntryLineToNode(
        CFGs, filePathList)  # for cfg entry nodes:{line:nodeid}

    for key in entryDict:  # add entry nodeid to line info
        CFGsNodeToLineDict[entryDict[key]] = key

    print("start - extracting apis...")
    apiLines = extractAPILines(doneLines, CFGs, sensiAPIPath,
                               CFGsNodeToLineDict)
    # apiLines = extractOperatorLines(doneLines, operatorLineDir, filePathList)
    print("end - extracting apis...")

    if (apiLines == list()):
        return dict()
    # =====================================#
    # build program dependence graph (PDG) #
    # =====================================#
    print("start - building PDG...")

    PDG = nx.DiGraph()
    control_edges = list()
    print("start - building CFGLineDepdcy...")
    CFGDepdcy, CFGLineDepdcy, CFGInfluence, CFGLineInfluence = buildCFGDepedcyInflueceDict(
        cfgPath, filePathList)  # {line:its cfg-dependency-line}
    for line in CFGLineDepdcy:
        control_dependent_line_set = CFGLineDepdcy[line]
        for control_dependent_line in control_dependent_line_set:
            control_edges.append((control_dependent_line, line, {"c/d": "c"}))
    PDG.add_edges_from(control_edges)
    print("end - building CFGLineDepdcy...")

    print("start - building VFGLineDepdcy...")
    SVFGnodeToLineDict = buildSVFGNodeToLineDict(SVFG, filePathList)
    value_edges = list()
    for n in SVFG._pred:
        if n not in SVFGnodeToLineDict:
            continue
        cur_line = SVFGnodeToLineDict[n]
        pred_lines = set()
        visited_nodeid = set()
        end = n.find(":s")
        if end != -1:
            n = n[:end]
        visited_nodeid.add(n)

        dfs_pred_lines(True, n, SVFGnodeToLineDict, SVFG._pred, pred_lines,
                       visited_nodeid)
        for pred_line in pred_lines:
            if pred_line != cur_line:
                value_edges.append((pred_line, cur_line, {"c/d": "d"}))
    PDG.add_edges_from(value_edges)

    print("end - building VFGLineDepdcy...")
    print("end - building PDG...")
    codeGadgetLinesDict = dict()  # {api:cdg}
    for apiLine in apiLines:
        print("start - processing apiline{}:...".format(apiLine))
        sliced_lines = set()

        # backward traversal
        bqueue = list()
        visited = set()
        bqueue.append(apiLine)
        visited.add(apiLine)
        while bqueue:
            fro = bqueue.pop(0)
            sliced_lines.add(fro)
            if fro in PDG._pred:
                for pred in PDG._pred[fro]:
                    if pred not in visited:
                        visited.add(pred)
                        bqueue.append(pred)

        # forward traversal
        fqueue = list()
        visited = set()
        fqueue.append(apiLine)
        visited.add(apiLine)
        while fqueue:
            fro = fqueue.pop(0)
            sliced_lines.add(fro)
            if fro in PDG._succ:
                for succ in PDG._succ[fro]:
                    if succ not in visited:
                        visited.add(succ)
                        fqueue.append(succ)
        print("end - processing apiline{}:...".format(apiLine))
        entryToLinesDict = dict()
        for relatedLine in sliced_lines:

            entry = locateLineEntry(relatedLine, list(entryDict.keys()))
            if entry not in entryToLinesDict:
                entryToLinesDict[entry] = list()
            entryToLinesDict[entry].append(relatedLine)

        for entry in entryToLinesDict:
            entryToLinesDict[entry] = sorted(entryToLinesDict[entry])

        callChianList = getCallChainList(calleeToCallerDict,
                                         entryToLinesDict.keys())

        maxLen = 0
        lenToCallChainDict = dict()
        for callChian in callChianList:
            if len(callChian) > maxLen:
                maxLen = len(callChian)
            if len(callChian) not in lenToCallChainDict:

                lenToCallChainDict[len(callChian)] = list()
            lenToCallChainDict[len(callChian)].append(callChian)

        finalCallChain = list()
        finalCallChain.extend(lenToCallChainDict[maxLen][0])
        for callChain in lenToCallChainDict[maxLen]:
            for entry in callChain:
                if entry not in set(finalCallChain):
                    finalCallChain.append(entry)
        for lenn in lenToCallChainDict:
            if lenn != maxLen:
                for callChain in lenToCallChainDict[lenn]:
                    for entry in callChain:
                        if entry not in set(finalCallChain):
                            finalCallChain.append(entry)

        codeGadgetLines = list()
        for entry in finalCallChain:
            codeGadgetLines.extend(entryToLinesDict[entry])

        codeGadgetLinesDict[apiLine] = codeGadgetLines

    return codeGadgetLinesDict


def generate_MULVDP_osp(config):
    '''
    apply open-source projects
    :param config:
    :param project: redis
    :return:
    '''
    data_root_dir = config.raw_data_folder
    sensiAPIPath = config.sensi_api_path
    noapiFile = config.no_api_path
    project = config.dataset.name
    dot_root_dir = join(data_root_dir, 'CVE', project, "dot")
    source_root_dir = join(data_root_dir, 'CVE', project, "source-code")
    source_code_dir = join(source_root_dir, project)
    # outputDir = rootDir + "json-raw3/"
    data_out_dir = join(config.data_folder, config.name, project)
    if (not os.path.exists(data_out_dir)):
        os.system(f"mkdir -p {data_out_dir}")
    cdg_out_path = join(data_out_dir, "all.txt")
    # operatorLineDir = data_root_dir + "key-operator/"
    xmlPath = join(source_root_dir, "manifest.xml")
    tree = ET.ElementTree(file=xmlPath)
    testcases = tree.findall("testcase")
    print("start - generating codeIDtoPath dict...")
    codeIDtoPath = getCodeIDtoPathDict(
        testcases, source_root_dir)  # {testcaseid:{filePath:set(vul lines)}}
    print("end - generating codeIDtoPath dict...")

    doneIDs = set()
    noApiIDs = set()
    if not os.path.exists(noapiFile):
        os.system("touch {}".format(noapiFile))
    with open(noapiFile, "r", encoding="utf-8") as f:
        noApiIDs = set(f.read().split(","))
    if not os.path.exists(join(data_out_dir, "doneID.txt")):
        os.system("touch {}".format(join(data_out_dir, "doneID.txt")))
    with open(join(data_out_dir, "doneID.txt"), "r", encoding="utf-8") as f:
        doneIDs = set(f.read().split(","))

    # =========================================================#
    # walk the testcase dir(subdirs : [testcaseid/dot files]) #
    # =========================================================#

    for root, testcaseIDList, files in os.walk(dot_root_dir):

        for testcaseID in testcaseIDList:
            if testcaseID in noApiIDs or testcaseID in doneIDs:
                continue
            #切换repo版本
            git_r = checkout_to(source_code_dir, testcaseID, project)
            filePathList = list(codeIDtoPath[testcaseID].keys())

            dotRoot = root + "/" + testcaseID + "/"
            svfgPath = dotRoot + "svfg_final.dot"
            cfgPath = dotRoot + "icfg_final.dot"
            callgPath = dotRoot + "callgraph_final.dot"
            print("start - processing files {} id {}...".format(
                filePathList, testcaseID))
            codeGadgetLinesDict = buildCodeGadgetList(set(), svfgPath, cfgPath,
                                                      sensiAPIPath,
                                                      filePathList, callgPath)

            if (codeGadgetLinesDict == dict()):
                print("no codeGadget!!end - processing files {}...".format(
                    filePathList))
                with open(noapiFile, 'a', encoding="utf-8") as f:
                    f.write(str(testcaseID))
                    f.write(",")
                continue

            print("end - processing files {}...".format(filePathList))

            fileContents = list()
            for filePath in filePathList:
                with open(join(source_code_dir, filePath),
                          "r",
                          encoding="utf-8",
                          errors="ignore") as f:
                    fileContents.append(f.readlines())

            for apiLine in codeGadgetLinesDict:
                codeGadgetLines = codeGadgetLinesDict[apiLine]
                if (len(codeGadgetLines) < 3):
                    continue
                apiLineSplit = apiLine.split("_")
                filePath = filePathList[int(apiLineSplit[0])]
                apiLinee = apiLineSplit[1]
                info = str(testcaseID) + " " + filePath + " " + apiLinee + "\n"
                content = ""
                target = 0  # correct
                for line in codeGadgetLines:
                    lineSplit = line.split("_")
                    fileidx = int(lineSplit[0])
                    fileContent = fileContents[fileidx]
                    content = content + fileContent[int(lineSplit[1]) -
                                                    1].strip() + "\n"
                    if target == 1:
                        continue
                    if (int(lineSplit[1]) in codeIDtoPath[testcaseID][
                            filePathList[fileidx]]):
                        target = 1

                with open(cdg_out_path, "a", encoding="utf-8",
                          errors="ignore") as f:
                    f.write(info + content + str(target) + "\n" +
                            "---------------------------------" + "\n")
                with open(join(data_out_dir, "doneID.txt"),
                          'a',
                          encoding="utf-8") as f:
                    f.write(str(testcaseID))
                    f.write(",")
            checkout_back(git_r, project)
    print("[program done!]")
    return


def generate_MULVDP(config):
    '''
    TODO: open-source projects
    :param config:
    :return:
    '''
    data_root_dir = config.raw_data_folder
    sensiAPIPath = config.sensi_api_path
    noapiFile = config.no_api_path
    cwEid = config.dataset.name
    dot_root_dir = join(data_root_dir, 'CWE', cwEid, "sard", "dot")
    if (len(os.listdir(dot_root_dir)) == 0):
        print("no dot file!")
        return
    source_root_dir = join(data_root_dir, 'CWE', cwEid, "sard", "source-code")
    # outputDir = rootDir + "json-raw3/"
    data_out_dir = join(config.data_folder, config.name, cwEid)
    if (not os.path.exists(data_out_dir)):
        os.system(f"mkdir -p {data_out_dir}")
    cdg_out_path = join(data_out_dir, "all.txt")
    # operatorLineDir = data_root_dir + "key-operator/"
    xmlPath = join(source_root_dir, "manifest.xml")
    tree = ET.ElementTree(file=xmlPath)
    testcases = tree.findall("testcase")
    print("start - generating codeIDtoPath dict...")
    codeIDtoPath = getCodeIDtoPathDict(
        testcases, source_root_dir)  # {testcaseid:{filePath:set(vul lines)}}
    print("end - generating codeIDtoPath dict...")

    doneIDs = set()
    noApiIDs = set()
    if not os.path.exists(noapiFile):
        os.system("touch {}".format(noapiFile))
    with open(noapiFile, "r", encoding="utf-8") as f:
        noApiIDs = set(f.read().split(","))
    if not os.path.exists(join(data_out_dir, "doneID.txt")):
        os.system("touch {}".format(join(data_out_dir, "doneID.txt")))
    with open(join(data_out_dir, "doneID.txt"), "r", encoding="utf-8") as f:
        doneIDs = set(f.read().split(","))

    # =========================================================#
    # walk the testcase dir(subdirs : [testcaseid/dot files]) #
    # =========================================================#

    for root, testcaseIDList, files in os.walk(dot_root_dir):

        for testcaseID in testcaseIDList:
            if testcaseID in noApiIDs or testcaseID in doneIDs:
                continue
            filePathList = list(codeIDtoPath[testcaseID].keys())

            dotRoot = root + "/" + testcaseID + "/"
            svfgPath = dotRoot + "svfg_final.dot"
            cfgPath = dotRoot + "icfg_final.dot"
            callgPath = dotRoot + "callgraph_final.dot"
            print("start - processing files {} id {}...".format(
                filePathList, testcaseID))
            codeGadgetLinesDict = buildCodeGadgetList(set(), svfgPath, cfgPath,
                                                      sensiAPIPath,
                                                      filePathList, callgPath)

            if (codeGadgetLinesDict == dict()):
                print("no codeGadget!!end - processing files {}...".format(
                    filePathList))
                with open(noapiFile, 'a', encoding="utf-8") as f:
                    f.write(str(testcaseID))
                    f.write(",")
                continue

            print("end - processing files {}...".format(filePathList))

            fileContents = list()
            for filePath in filePathList:
                with open(join(source_root_dir, filePath),
                          "r",
                          encoding="utf-8",
                          errors="ignore") as f:
                    fileContents.append(f.readlines())

            for apiLine in codeGadgetLinesDict:
                codeGadgetLines = codeGadgetLinesDict[apiLine]
                if (len(codeGadgetLines) < 3):
                    continue
                apiLineSplit = apiLine.split("_")
                filePath = filePathList[int(apiLineSplit[0])]
                apiLinee = apiLineSplit[1]
                info = str(testcaseID) + " " + filePath + " " + apiLinee + "\n"
                content = ""
                target = 0  # correct
                for line in codeGadgetLines:
                    lineSplit = line.split("_")
                    fileidx = int(lineSplit[0])
                    fileContent = fileContents[fileidx]
                    content = content + fileContent[int(lineSplit[1]) -
                                                    1].strip() + "\n"
                    if target == 1:
                        continue
                    if (int(lineSplit[1]) in codeIDtoPath[testcaseID][
                            filePathList[fileidx]]):
                        target = 1

                with open(cdg_out_path, "a", encoding="utf-8",
                          errors="ignore") as f:
                    f.write(info + content + str(target) + "\n" +
                            "---------------------------------" + "\n")
                with open(join(data_out_dir, "doneID.txt"),
                          'a',
                          encoding="utf-8") as f:
                    f.write(str(testcaseID))
                    f.write(",")
    print("[program done!]")
    return
