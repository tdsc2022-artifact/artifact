from typing import List, Dict, Set
import os
import xml.etree.ElementTree as ET
from utils.common import CWEID_ADOPT
from utils.git_checkout import checkout_to, checkout_back, checkout_to_pre
import shutil
def getCodeIDtoPathDict(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    '''build code testcaseid to path map

    use the manifest.xml. build {testcaseid:{filePath:set(vul lines)}}
    filePath use relevant path, e.g., CWE119/cve/source-code/project_commit/...
    :param testcases:
    :return: {testcaseid:{filePath:set(vul lines)}}
    '''
    codeIDtoPath: Dict[str, Dict[str, Set[int]]] = {}
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

def getCodeIDtoPathDict_osp(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    '''build code testcaseid to path map

    use the manifest.xml. build {testcaseid:{filePath:set(vul lines)}}
    filePath use relevant path, e.g., CWE119/cve/source-code/project_commit/...
    :param testcases:
    :return: {testcaseid:{filePath:set(vul lines)}}
    '''
    codeIDtoPath_1: Dict[str, Dict[str, List[int]]] = {}
    codeIDtoPath_0: Dict[str, Dict[str, List[int]]] = {}
    for testcase in testcases:
        files = testcase.findall("file")
        testcaseid = testcase.attrib["id"]
        codeIDtoPath_1[testcaseid] = dict()
        codeIDtoPath_0[testcaseid] = dict()
        for file in files:
            path = file.attrib["path"]
            flaws = file.findall("flaw")
            # print(mixeds)
            VulLine = set()
            SafeLine = set()
            if (flaws != []):
                # targetFilePath = path
                if (flaws != []):
                    for flaw in flaws:
                        flaw_type = flaw.attrib["type"]
                        if flaw_type == '1':
                            VulLine.add(int(flaw.attrib["line"]))
                        elif flaw_type == '0':
                            SafeLine.add(int(flaw.attrib["line"]))
            if len(VulLine) > 0:  
                VulLine = list(VulLine)
                codeIDtoPath_1[testcaseid][path] = dict()          
                codeIDtoPath_1[testcaseid][path]['VulLine'] = VulLine
            if len(SafeLine) > 0:
                SafeLine = list(SafeLine)
                codeIDtoPath_0[testcaseid][path] = dict()          
                codeIDtoPath_0[testcaseid][path]['SafeLine'] = SafeLine
    return codeIDtoPath_0, codeIDtoPath_1

def getCodeIDtoPathDict_d2a(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    codeIDtoPath: Dict[str, Dict[str, List[int]]] = {}
    for testcase in testcases:
        files = testcase.findall("file")
        testcaseid = testcase.attrib["id"]
        codeIDtoPath[testcaseid] = dict()
        for file in files:
            path = file.attrib["path"]
            flaws = file.findall("flaw")
            # print(mixeds)
            VulLine = list()
            Vul_bugtype = list()
            SafeLine = list()
            Safe_bugtype = list()
            if (flaws != []):
                # targetFilePath = path
                if (flaws != []):
                    for flaw in flaws:
                        flaw_type = flaw.attrib["type"]
                        if flaw_type == '1':
                            VulLine.append(flaw.attrib["line"])
                            Vul_bugtype.append(flaw.attrib["bug_type"])
                        elif flaw_type == '0':
                            SafeLine.append(flaw.attrib["line"])
                            Safe_bugtype.append(flaw.attrib["bug_type"])
            codeIDtoPath[testcaseid][path] = dict()      
            codeIDtoPath[testcaseid][path]['VulLine'] = dict()    
            codeIDtoPath[testcaseid][path]['SafeLine'] = dict()         
            if len(VulLine) > 0:  
                for line, bug_type in zip(VulLine, Vul_bugtype):
                    if line not in codeIDtoPath[testcaseid][path]['VulLine'].keys():
                        codeIDtoPath[testcaseid][path]['VulLine'][line] = list()
                    bug_types = codeIDtoPath[testcaseid][path]['VulLine'][line]
                    bug_types.append(bug_type)
                    codeIDtoPath[testcaseid][path]['VulLine'][line] = list(set(bug_types))
            if len(SafeLine) > 0:     
                for line, bug_type in zip(SafeLine, Safe_bugtype):
                    if line not in codeIDtoPath[testcaseid][path]['SafeLine'].keys():
                        codeIDtoPath[testcaseid][path]['SafeLine'][line] = list()
                    bug_types = codeIDtoPath[testcaseid][path]['SafeLine'][line]
                    bug_types.append(bug_type)
                    codeIDtoPath[testcaseid][path]['SafeLine'][line] = list(set(bug_types))

    return codeIDtoPath
    


def create_d2a_source_code(d2a_data_root_dir, output_dir):
    """
    @description  :
                Extract the source code of each version
                according to the commit_id in the xml
    ---------
    @param  :d2a_data_root_dir: the dir of all project source-code
             output_dir: the output dir of extracted source code
    -------
    @Returns  :
    -------
    """
    
    
    for cve_project in os.listdir(d2a_data_root_dir):
        try:
            source_code_root = os.path.join(d2a_data_root_dir, cve_project, 'source-code', cve_project)
            xmlPath = os.path.join(d2a_data_root_dir, cve_project, 'source-code', 'manifest_btp.xml')
            changed_source_code_dir = os.path.join(output_dir, cve_project)
            if not os.path.exists(changed_source_code_dir):
                os.makedirs(changed_source_code_dir)
            tree = ET.ElementTree(file=xmlPath)
            testcases = tree.findall("testcase")
            for testcase in testcases:
                files = testcase.findall("file")
                testcaseid = testcase.attrib["id"]
                #切换到testcase版本
                repo_dir = source_code_root
                git_r = checkout_to(repo_dir, testcaseid, cve_project)
                for file in files:
                    path = file.attrib["path"]
                    flaws = file.findall("flaw")
                    if len(flaws) > 0:
                        code_dir = os.path.join(changed_source_code_dir,testcaseid)
                        if not os.path.exists(code_dir):
                            os.mkdir(code_dir)
                        code = os.path.join(source_code_root, path)
                        p,f = os.path.split(path)
                        code_path = os.path.join(code_dir,p)
                        if not os.path.exists(code):
                            continue
                        if not os.path.exists(code_path):
                            os.makedirs(code_path)
                        shutil.copyfile(code, os.path.join(code_dir,path))
                checkout_back(git_r, cve_project)
        except Exception as e:
            with open('create_d2a_code.log','a+') as f:
                f.write(cve_project+'\n')
                f.write(str(e)+'\n')
                f.close()


