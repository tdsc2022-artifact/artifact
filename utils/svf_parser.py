'''svf-related graph parser

'''
import os
import json


def getNodeIDNo(NodeID, G):
    '''nodeid -> nodeidNo

    nodeid mean the id in dot, nodeidNo mean the id in the real node
    if no nodeidNo, return the nodeid itself

    :param NodeID:
    :param G:
    :return:
    '''
    node = G._node[NodeID]
    if ("label" in node.keys()):
        label = node["label"]
        end = label.find("\\n")
        if end != -1:
            return int(label[10:end])
    return NodeID


def buildCFGNodeToLineDict(CFGs, filePathList):
    '''nodeid->line(do not calc entry block!)

    nodeid is the address in dot file

    :param CFGs:
    :return: {nodeid:line(int)}
    '''

    nodeToLineDict = dict()

    for key in CFGs._node:
        node = CFGs._node[key]
        if "label" in node.keys():
            label = node['label']
            start = label.find("ln:")
            end = label.find("fl:")
            for filePathidx in range(len(filePathList)):
                filePath = filePathList[filePathidx]
                if label.find(filePath) != -1:

                    if (start != -1):
                        line = label[start + 4:end - 1]
                        if (line != "0"):
                            nodeToLineDict[key] = str(filePathidx) + "_" + line
                    break

    return nodeToLineDict


def locateLineEntry(line, entries):
    '''locate which method line the line belongs to

    :param line:
    :param entries:
    :return:
    '''
    lineSplit = line.split("_")
    lineEntries = list()
    for entry in entries:
        entrySplit = entry.split("_")
        if entrySplit[0] == lineSplit[0]:
            lineEntries.append(int(entrySplit[1]))
    if len(lineEntries) == 0:
        return None
    lineEntries.sort()
    result = lineEntries[-1]
    lineInt = int(lineSplit[1])
    for entryidx in range(1, len(lineEntries)):
        if lineInt < lineEntries[entryidx]:
            result = lineEntries[entryidx - 1]
            break

    return lineSplit[0] + "_" + str(result)


def getCFGFuncEntryExit(CFGs, filePathList):
    '''get all functions in CFGs

    build a cfg method dict. see the return for detail

    :param CFGs:
    :param filePathList:
    :return:
        :first {funcname:{"entry": , "entryLine": , "exit": , "exitLine":}}
        :second {entryLine:funcname}
    '''
    funcDict = dict()
    lineToEntry = dict()

    for key in CFGs._node:
        node = CFGs._node[key]
        if "color" in node.keys():
            if node["color"] == "yellow":  # entry
                label = node["label"]
                for filePathidx in range(len(filePathList)):
                    filePath = filePathList[filePathidx]
                    if label.find(filePath) != -1:
                        Linestart = label.find("line:")
                        LineEnd = label.find("file:")
                        FunStart = label.find("Fun[")
                        FunEnd = label.find("]")
                        line = None  # may not have a line
                        if Linestart != -1:
                            line = label[Linestart + 6:LineEnd - 1]
                        if FunStart != -1 and line != "0" and line is not None:  # line should not be 0
                            funcName = label[FunStart + 4:FunEnd]
                            if funcName not in funcDict:
                                funcDict[funcName] = dict()
                            # print(funcName)
                            funcDict[funcName]["entry"] = key
                            funcDict[funcName]["entryLine"] = str(
                                filePathidx) + "_" + line
                            lineToEntry[str(filePathidx) + "_" +
                                        line] = funcName
                        break

    for key in CFGs._node:
        node = CFGs._node[key]
        if "color" in node.keys():
            if node["color"] == "green":  # exit
                label = node["label"]

                Linestart = label.find("ln:")
                LineEnd = label.find("fl:")
                FunStart = label.find("Fun[")
                FunEnd = label.find("]")
                line = None  # may not have a line
                if Linestart != -1:
                    line = label[Linestart + 4:LineEnd - 1]
                if FunStart != -1 and line != "0":  # line should not be 0
                    funcName = label[FunStart + 4:FunEnd]
                    if funcName in funcDict:  # may not have filePath info
                        funcDict[funcName]["exit"] = key
                        funcDict[funcName]["exitLine"] = (
                            funcDict[funcName]["entryLine"].split("_")[0] +
                            "_" + line) if line is not None else None

    return funcDict, lineToEntry


def getCFGsEntryLineToNode(CFGs, filePathList):
    '''get CFGs' all entries - method to entry

    build a cfg method dict. see the return for detail.

    :param CFGs:
    :param filePathList: source code
    :return: {entryline:nodeid}
    '''
    entryDict = dict()

    for key in CFGs._node:
        node = CFGs._node[key]
        if "color" in node.keys():
            if node["color"] == "yellow":
                label = node["label"]
                start = label.find("line")
                end = label.find("file")
                for filePathidx in range(len(filePathList)):
                    filePath = filePathList[filePathidx]
                    if label.find(filePath) != -1:
                        if start != -1 and end != -1:
                            if (label[start + 6:end - 1] != "0"):
                                entryDict[str(filePathidx) + "_" +
                                          label[start + 6:end - 1]] = key
                        break

    return entryDict


def buildSVFGLineToNodeDict(SVFG, filePathList):
    '''line->[nodeid] may be multiple

    build a svfg node line to list of nodeid(node address in dot file) dict.

    :param SVFG:
    :return: {line(int):[nodeid]}
    '''
    lineToNodeDict = dict()

    for key in SVFG._node:
        node = SVFG._node[key]
        if "label" in node.keys():
            label = node['label']
            start = label.find("ln:")
            end = label.find("fl:")
            for filePathidx in range(len(filePathList)):
                filePath = filePathList[filePathidx]
                if label.find(filePath) != -1:
                    if (start != -1):

                        line = str(filePathidx) + "_" + label[start + 4:end -
                                                              1]
                        if line in lineToNodeDict.keys():
                            lineToNodeDict[line].append(key)
                        else:
                            lineToNodeDict[line] = [key]

                    start = label.find("line:")
                    end = label.find("file:")
                    if (start != -1):

                        line = str(filePathidx) + "_" + label[start + 6:end -
                                                              1]
                        if line in lineToNodeDict.keys():
                            lineToNodeDict[line].append(key)
                        else:
                            lineToNodeDict[line] = [key]
                    break

    return lineToNodeDict


def buildSVFGNodeToLineDict(SVFG, filePathList):
    '''nodeid->line(do not calc entry block!)

    build a svfg nodeid(node address in dot file) to node line dict.

    :param SVFG:
    :return: {nodeid:line(int_int)}
    '''

    nodeToLineDict = dict()

    for key in SVFG._node:
        node = SVFG._node[key]
        if "label" in node.keys():
            label = node['label']
            start = label.find("ln:")
            end = label.find("fl:")
            for filePathidx in range(len(filePathList)):
                filePath = filePathList[filePathidx]
                if label.find(filePath) != -1:
                    if label[start + 4:end - 1] != "0":
                        if (start != -1):
                            line = str(filePathidx) + "_" + label[start +
                                                                  4:end - 1]

                            nodeToLineDict[key] = line
                    start = label.find("line:")
                    end = label.find("file:")
                    if label[start + 6:end - 1] != "0":
                        if (start != -1):
                            line = str(filePathidx) + "_" + label[start +
                                                                  6:end - 1]
                            nodeToLineDict[key] = line
                    break
    return nodeToLineDict


def file_len(fname):
    '''
    file line number
    :param fname:
    :return:
    '''
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        i = -1
        for i, l in enumerate(f):
            pass
        return i + 1


def buildCallGraphDict(CFGs, CallGraph, filePathList):
    '''build callgraph

    Use callgraph_final.dot. First find the root caller.
    Then use dfs to build the call path. Note that shape and funcName is important for judging
    whether the method is user-defined.

    :param CFGs: CFG in networkx
    :param CallGraph: callgraph in networkx
    :return: {callee line:set(caller line)}
    '''
    callDict = dict()
    funcNameToLine = dict()
    for key in CFGs._node:
        node = CFGs._node[key]
        if "color" in node.keys():
            if node["color"] == "yellow":
                label = node["label"]
                start = label.find("line")
                end = label.find("file")
                for filePathidx in range(len(filePathList)):
                    filePath = filePathList[filePathidx]
                    if label.find(filePath) != -1:

                        if start != -1 and end != -1:
                            line = label[start + 6:end - 1]
                            start = label.find("Fun[")

                            if start != -1 and line != "0":
                                funcName = label[start + 4:-3]
                                funcNameToLine[funcName] = str(
                                    filePathidx) + "_" + line
                        break

    CallGraphEntrys = list()
    for CallGraphNodeid in CallGraph._node:
        CallGraphNodePreds = CallGraph._pred[CallGraphNodeid]
        if CallGraphNodePreds == dict():  # no pred caller - root caller
            CallGraphEntrys.append(CallGraphNodeid)

    for CallGraphEntry in CallGraphEntrys:
        entryNode = CallGraph._node[CallGraphEntry]
        if "label" in entryNode.keys():
            label = entryNode["label"]
            left = label.find("{")
            right = label.find("}")
            if "shape" in entryNode.keys(
            ) and entryNode["shape"] == "circle":  # user-defined method
                entryFuncName = label[left + 1:right]
                if entryFuncName in funcNameToLine:
                    entryFuncLine = funcNameToLine[entryFuncName]
                    Visited = set()
                    Visited.add(CallGraphEntry)
                    dfsBuildCallGraphDict(callDict, Visited, CallGraphEntry,
                                          entryFuncLine, CallGraph,
                                          funcNameToLine)

    return callDict


def dfsBuildCallGraphDict(callDict, Visited, curNodeid, curFuncLine, CallGraph,
                          funcNameToLine):
    '''key method - buildCallGraphDict

    dfs from the root node. find the call chain

    :param callDict: the result {callee line:set(caller line)}
    :param Visited:
    :param curNodeid:
    :param curFuncLine:
    :param CallGraph: callgraph in networkx
    :param funcNameToLine:
    :return: None
    '''

    nextNodeIDs = CallGraph._succ[curNodeid]

    for nextNodeID in nextNodeIDs:
        nextNode = CallGraph._node[nextNodeID]
        if "label" in nextNode.keys():
            label = nextNode["label"]
            left = label.find("{")
            right = label.find("}")
            if "shape" in nextNode.keys() and nextNode["shape"] == "circle":
                nextFuncName = label[left + 1:right]
                # if nextFuncName == '_ZNSt7__cxx114listIPlSaIS1_EE4backEv' or nextFuncName == '__clang_call_terminate' or nextFuncName == '__cxx_global_var_init' or nextFuncName == '_ZNSt7__cxx114listImSaImEE4backEv':
                #     continue
                if nextFuncName not in funcNameToLine:
                    continue
                nextFuncLine = funcNameToLine[nextFuncName]
                if nextFuncLine in callDict.keys():
                    callDict[nextFuncLine].add(curFuncLine)
                else:
                    callDict[nextFuncLine] = set()
                    callDict[nextFuncLine].add(curFuncLine)

                if nextNodeID not in Visited:
                    Visited.add(nextNodeID)
                    dfsBuildCallGraphDict(callDict, Visited, nextNodeID,
                                          nextFuncLine, CallGraph,
                                          funcNameToLine)


def extractOperatorLines(doneLines, operatorLineDir, filePathList):
    '''

    :param doneLines:
    :param operatorLineDir:
    :return:
    '''
    operatorLineList = list()
    if os.path.exists(operatorLineDir):

        with open(operatorLineDir, "r", encoding="utf-8",
                  errors="ignore") as f:
            load = json.load(f)
            for file in load:
                if len(file) > 1:

                    for filePathidx in range(len(filePathList)):
                        if filePathList[filePathidx] == file[0]:
                            for line in file[1:]:
                                print("!!!!operator", line)
                                line = str(filePathidx) + "_" + str(line)
                                if line not in doneLines:

                                    operatorLineList.append(line)
                            break
    return operatorLineList


def extractAPILines(doneLines, CFGs, sensiAPIPath, CFGsNodeToLineDict):
    '''extract sensitive api-lines in CFGs

    Use apis in sensiAPIPath. API in the cfgs is in yellow node.
    The api call line is the pred node of the yellow node.

    :param CFGs: CFG in networkx
    :param sensiAPIPath: sensitive api path
    :param CFGsNodeToLineDict {node:line}
    :return: sensitive api-lines
    '''
    with open(sensiAPIPath, "r", encoding="utf-8") as f:
        sensiAPIList = set(f.read().split(","))
        # sensitive api set, for judge whether the api in the CFGs is sensitive
        sensiAPISet = set()
        for sensiAPI in sensiAPIList:
            sensiAPISet.add(sensiAPI.strip())

    # print(sensiAPIList)
    apilines = list()

    for key in CFGs._node:
        node = CFGs._node[key]
        if "color" not in node.keys():
            continue
        if node['color'] == "yellow":
            if "label" in node:
                label = node["label"]
                start = label.find("Fun[")
                end = label.find("]")
                if start != -1:
                    isApi = False
                    funName = node["label"][start + 4:end]
                    apiNameL = funName.split(".")
                    apiNameL.append(funName)
                    for apiName in apiNameL:

                        if apiName in sensiAPISet:
                            isApi = True
                            break

                    if not isApi and (key not in CFGsNodeToLineDict) and len(
                            CFGs._succ[key]) == 0:
                        isApi = True

                    if isApi:

                        preds = CFGs._pred[key]
                        for pred in preds:
                            if pred[:-3] in CFGsNodeToLineDict:
                                if CFGsNodeToLineDict[
                                        pred[:-3]] not in doneLines:

                                    apilines.append(
                                        CFGsNodeToLineDict[pred[:-3]])
                                else:
                                    print(
                                        "!!!!!!!!!!!!!apiLine{} has processed".
                                        format(CFGsNodeToLineDict[pred[:-3]]))

    return apilines


def getCallChainList(calleeToCallerDict, entries):
    '''

    :param calleeToCallerDict:
    :param entries:
    :return:
    '''
    callChainList = list()
    for entry in list(entries):
        callChain = list()
        callChain.insert(0, entry)
        dfsCallChain(entry, calleeToCallerDict, entries, callChainList,
                     callChain)
    return callChainList


def dfsCallChain(entry, calleeToCallerDict, entries, callChainList, callChain):
    '''

    :param entry:
    :param calleeToCallerDict:
    :param entries:
    :param callChainList:
    :param callChain:
    :return:
    '''
    if entry not in calleeToCallerDict:
        callChainList.append(callChain.copy())
        return
    callerList = calleeToCallerDict[entry]
    appendOnce = False
    for caller in callerList:
        if caller in callChain:
            if not appendOnce:
                appendOnce = True
                callChainList.append(callChain.copy())
            continue
        if caller in entries:

            callChain.insert(0, caller)
            dfsCallChain(caller, calleeToCallerDict, entries, callChainList,
                         callChain)
            callChain.pop(0)
            return
        else:
            if not appendOnce:
                appendOnce = True
                callChainList.append(callChain.copy())

    return


def dfs_pred_lines(is_begin, cur, SVFGnodeToLineDict, preds, pred_lines,
                   visited_nodeid):

    if (not is_begin and cur in SVFGnodeToLineDict):
        pred_lines.add(SVFGnodeToLineDict[cur])
        return

    for pred in preds[cur]:
        end = pred.find(":s")
        if end != -1:
            pred = pred[:end]
        if pred not in visited_nodeid:
            visited_nodeid.add(pred)
            dfs_pred_lines(False, pred, SVFGnodeToLineDict, preds, pred_lines,
                           visited_nodeid)