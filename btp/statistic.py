import os
import xml.etree.ElementTree as ET
import json
import sys
sys.path.append("..")
# from preprocessing.token_preprocess import statistic_samples as token_SS
from preprocessing.sys_preprocess import statistic_samples as sys_SS
from preprocessing.cdg_preprocess import statistic_samples as vul_SS
from utils.common import get_config
def statistic_samples_from_xml(root):
    data = dict()
    all_bug_type_count = 0
    bug_type_count = dict()
    all_count_0 = 0
    all_count_1 = 0
    for project in os.listdir(root):
        if project == 'libtiff':
            continue
        source_root_dir = os.path.join(root, project, 'source-code')
        xmlPath = os.path.join(source_root_dir, "manifest.xml")
        tree = ET.ElementTree(file=xmlPath)
        testcases = tree.findall("testcase")
    # project{
    #     'bug_type':{
    #         '0':count,
    #         '1':count,
    #     }
    # }
        data[project] = dict()
        project_bug_type_count = 0
        project_count_0 = 0
        project_count_1 = 0
        for testcase in testcases:
            files = testcase.findall("file")
            for file in files:
                flaws = file.findall("flaw")
                for flaw in flaws:
                    flaw_type = flaw.attrib["type"]
                    bug_type = flaw.attrib["bug_type"]
                    if bug_type not in bug_type_count.keys():
                        bug_type_count[bug_type] = dict()
                    
                    if bug_type not in data[project].keys():
                        data[project][bug_type] = dict()
                        project_bug_type_count += 1
                    if flaw_type == '0':
                        project_count_0 += 1
                        if '0' not in bug_type_count[bug_type].keys():
                            bug_type_count[bug_type]['0'] = 0
                        bug_type_count[bug_type]['0'] = bug_type_count[bug_type]['0'] +1 
                        if '0' not in data[project][bug_type].keys():
                            data[project][bug_type]['0'] = 0
                        data[project][bug_type]['0'] = data[project][bug_type]['0'] + 1
                    else:
                        project_count_1 += 1
                        if '1' not in bug_type_count[bug_type].keys():
                            bug_type_count[bug_type]['1'] = 0
                        bug_type_count[bug_type]['1'] = bug_type_count[bug_type]['1'] +1 
                        if '1' not in data[project][bug_type].keys():
                            data[project][bug_type]['1'] = 0
                        data[project][bug_type]['1'] = data[project][bug_type]['1'] + 1
        data[project]['project_bug_type_count'] = project_bug_type_count
        data[project]['project_count_0'] = project_count_0
        data[project]['project_count_1'] = project_count_1
        all_count_0 += project_count_0
        all_count_1 += project_count_1
    bug_type_count['all_bug_type_count'] = len(bug_type_count.keys())
    data['bug_type_count'] = bug_type_count
    data['all_count_0'] = all_count_0
    data['all_count_1'] = all_count_1
    with open('statistic.json', 'w') as f:
        json.dump(data, f, indent=2)
        f.close()
def statistic_samples_from_alltxt(method_name, dataset_name):
    
    config = get_config(method_name, dataset_name)
    function = {
        "sysevr":sys_SS,
        "vuldeepecker":vul_SS,
        # "token":token_SS
    }
    function[method_name](config)
   