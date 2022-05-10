import sys
import json
import gzip
import pickle
import base64
import os
import jsonpath
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import json
from os import path
sys.path.append("..")
from joern_slicer.slicing import get_graph_d2a, get_slice_d2a
from utils.plot_result import autolabel


def saveXML(xmltree, filename, indent="\t", newl="\n", encoding="utf-8"):
    """
    @description  : Formatted output xml
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    rawText = ET.tostring(xmltree)
    dom = minidom.parseString(rawText)
    with open(filename, 'w') as f:
        dom.writexml(f, "", indent, newl, encoding)


def json_to_xml(project, root, output_dir):
    """
    @description  : Organize D2A data in json format in xml file format as follows
                    <commit_id>
                        <path>
                            <line>
                                other information
                            </line>
                        </path>
                    </commit_id>
    
    ---------
    @param  :   project: project name;
                root:D2A json dir;
                output_dir:xml output dir
    -------
    @Returns  :
    -------
    """
    output = os.path.join(output_dir, project,
                          'source-code', 'manifest_btp.xml')
    json_file_path = [project + '_labeler_1.json']
    all_samples = dict()
    for path in json_file_path:
        with open(os.path.join(root, path), 'r') as f:
            json_list = json.load(f)
            f.close()
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
            if commit_id not in all_samples.keys():
                all_samples[commit_id] = dict()
            if adjusted_bug_loc != None :
                file_path = adjusted_bug_loc['file']
                line = adjusted_bug_loc['line']
            else :
                file_path = bug_info['file']
                line = bug_info['line']
            if file_path not in all_samples[commit_id].keys():
                all_samples[commit_id][file_path] = dict()
            if 'VulLine' not in all_samples[commit_id][file_path].keys():
                all_samples[commit_id][file_path]['VulLine'] = list()
            if 'Vul_bugtype' not in all_samples[commit_id][file_path].keys():
                all_samples[commit_id][file_path]['Vul_bugtype'] = list()
            if 'trace_list' not in all_samples[commit_id][file_path].keys():
                all_samples[commit_id][file_path]['trace_list'] = list()
            if 'functions_list' not in all_samples[commit_id][file_path].keys(
            ):
                all_samples[commit_id][file_path]['functions_list'] = list()
            VulLines = all_samples[commit_id][file_path]['VulLine']
            Vul_bugtype = all_samples[commit_id][file_path]['Vul_bugtype']
            trace_list = all_samples[commit_id][file_path]['trace_list']
            functions_list = all_samples[commit_id][file_path]['functions_list']
            VulLines.append(line)
            Vul_bugtype.append(bug_type)
            trace_list.append(trace)
            functions_list.append(functions)
            all_samples[commit_id][file_path]['VulLine'] = VulLines
            all_samples[commit_id][file_path]['Vul_bugtype'] = Vul_bugtype
            all_samples[commit_id][file_path]['trace_list'] = trace_list
            all_samples[commit_id][file_path]['functions_list'] = functions_list
    root_et = ET.Element("container")
    for commit_id in all_samples:
        sample = all_samples[commit_id]
        testcase = ET.Element('testcase')
        testcase.set('id', commit_id)
        testcase.set('type', 'Source Code')
        for file_path in sample:
            VulLines = sample[file_path]['VulLine']
            Vul_bugtype = sample[file_path]['Vul_bugtype']
            trace_list = sample[file_path]['trace_list']
            functions_list = sample[file_path]['functions_list']
            file = ET.Element("file")
            file.set('path', file_path)
            for line, bug_type, trace, functions in zip(
                    VulLines, Vul_bugtype, trace_list, functions_list):
                flaw = ET.Element('flaw')
                flaw.set('type', '1')
                flaw.set('line', str(line))
                flaw.set('bug_type', bug_type)
                trace_c = ET.Element('trace')
                functions_c = ET.Element('functions')
                for step in trace:
                    step_c = ET.Element('step')
                    index = step['idx']
                    step_file_path = step['file']
                    func_name = step['func_name']
                    loc = step['loc']
                    step_c.set('idx', str(index))
                    step_c.set('file', str(step_file_path))
                    step_c.set('func_name', str(func_name))
                    step_c.set('loc', loc)
                    trace_c.append(step_c)
                flaw.append(trace_c)
                for func_key in functions:
                    function = ET.Element('function')
                    func = functions[func_key]
                    func_file_path = func['file']
                    func_name = func['name']
                    loc = func['loc']
                    function.set('file', str(func_file_path))
                    function.set('func_name', str(func_name))
                    function.set('loc', str(loc))
                    functions_c.append(function)
                flaw.append(functions_c)
                file.append(flaw)
            testcase.append(file)
        root_et.append(testcase)
    saveXML(root_et, output)

    return all_samples






def btp(root_dir, projects, d2a_slice_root):
    """
    @description  : Categorize by vulnerability type, 
                    and respectively count the bug_trace_lines, method_lines, 
                    SySeVR_lines, VulDeepecker_lines in the D2A data
    
    ---------
    @param  :   project: project name;
                root_dir:The root directory where the xml files of all projects are located
                d2a_silce_root:The root directory where the d2a slice of all projects are located
    -------
    @Returns  :
    -------
    """

    # projects = ['libav']
    BUFFER_OVERRUN = ['BUFFER_OVERRUN_L1', 'BUFFER_OVERRUN_L2',
                        'BUFFER_OVERRUN_L3', 'BUFFER_OVERRUN_L4', 
                        'BUFFER_OVERRUN_L5', 'BUFFER_OVERRUN_S2',
                        'BUFFER_OVERRUN_U5']
    INTEGER_OVERFLOW = ['INTEGER_OVERFLOW_L1', 'INTEGER_OVERFLOW_L2',
                            'INTEGER_OVERFLOW_L5', 'INTEGER_OVERFLOW_R2',
                            'INTEGER_OVERFLOW_U5']
    NULL_DEREFERENCE = ['DANGLING_POINTER_DEREFERENCE',
                        'NULL_DEREFERENCE',
                        'NULLPTR_DEREFERENCE']   
    MEMOREY_LEAK = ['MEMORY_LEAK', 'PULSE_MEMORY_LEAK']  
    json_dict = {}
    for project in projects:
        xmlPath = os.path.join(root_dir, project,
                               'source-code', 'manifest_btp.xml')
        tree = ET.ElementTree(file=xmlPath)
        testcases = tree.findall("testcase")
        for testcase in testcases:
            testcase_id = testcase.attrib['id']
            slice_path = os.path.join(
                d2a_slice_root, 
                project, 
                testcase_id+'.json'
            )
            if os.path.exists(slice_path):
                with open(slice_path, 'r') as f:
                    slice_json = json.load(f)
                    f.close()
            else:
                slice_json = []
            file_info = testcase.findall('file')
            for info in file_info:
                
                path = info.attrib["path"]
                flaws = info.findall('flaw')
                # joern_parse_result_file = os.path.join(joern_parse_result_path+'/'+project+'/home/chengxiao/project/vul_detect/joern_slicer/d2a_resources_btp/'+project, testcase_id, path)
                
                # if os.path.isdir(joern_parse_result_file):
                #     print(joern_parse_result_file)
                # combined_graph, data_graph = get_graph_d2a(joern_parse_result_file)

                for flaw in flaws:
                    sub_bug_type = flaw.attrib['bug_type']
                    if sub_bug_type in BUFFER_OVERRUN:
                        bug_type = 'BUFFER_OVERRUN'
                    elif sub_bug_type in INTEGER_OVERFLOW:
                        bug_type = 'INTEGER_OVERFLOW'
                    elif sub_bug_type in NULL_DEREFERENCE:
                        bug_type = 'NULL_DEREFERENCE'
                    elif sub_bug_type in MEMOREY_LEAK:
                        bug_type = 'MEMOREY_LEAK'
                    else:
                        bug_type = sub_bug_type
                    print(sub_bug_type + ' ' + bug_type)
                    
                    if bug_type not in json_dict.keys():
                        json_dict[bug_type] = dict()
                        json_dict[bug_type]['bug_info'] = list()
                    bug_info_list = json_dict[bug_type]['bug_info']
                    bug_info = dict()
                    bug_info['path'] = path
                    line = int(flaw.attrib['line'])
                    functions = flaw.find('functions').findall('function')

                    #trace

                    trace = flaw.find('trace')
                    step = trace.findall('step')
                    trace_list = list()
                    for s in step:
                        trace_file_path = s.attrib['file']
                        trace_loc = s.attrib['loc']
                        trace_list.append(trace_file_path + ' ' +
                                          trace_loc.split(':')[0])
                    bug_info['trace_line'] = trace_list

                    #method
                    func_info = dict()
                    for func in functions:
                        func_path = func.attrib['file']
                        func_name = func.attrib['func_name']
                        loc = func.attrib['loc']
                        start = loc.split('-')[0]
                        end = loc.split('-')[1]
                        start_line = int(start.split(':')[0])
                        end_line = int(end.split(':')[0])
                        if (line <= end_line) and (line >= start_line) :
                            func_info['func_name'] = func_name
                            func_info['loc'] = loc
                            lines = list()
                            for i in range(start_line, end_line + 1):
                                lines.append(i)
                            func_info['lines'] = lines
                    bug_info['method_line'] = func_info

                    #slice
                    sy_lines = list()
                    vd_lines = list()

                    for slice_info in slice_json:
                        slice_path = '/'.join(slice_info['file_path'].split('/')[11:])         
                        if slice_path == path:
                            for key in slice_info:
                                if key.endswith('sy'):
                                    slice_list = slice_info[key]
                                    for s in slice_list:
                                        if line in s:
                                            sy_lines.extend(s)
                                elif key.endswith('vd'):
                                    slice_list = slice_info[key]
                                    for s in slice_list:
                                        if line in s:
                                            vd_lines.extend(s)
                    if len(sy_lines) > 0:
                        bug_info['sy_trace_lines'] = trace_list
                        bug_info['sy_lines'] = sy_lines
                    else:
                        bug_info['sy_trace_lines'] = []
                        bug_info['sy_lines'] = []
                    if len(vd_lines) > 0:
                        bug_info['vd_trace_lines'] = trace_list
                        bug_info['vd_lines'] = vd_lines
                    else:
                        bug_info['vd_trace_lines'] = []
                        bug_info['vd_lines'] = []
                    # data_instance = get_slice_d2a(combined_graph, data_graph, line)
                    # bug_info['sy_lines'] = data_instance['all_slices_sy']
                    # bug_info['vd_lines'] = data_instance['all_slices_vd']

                    # #cfg
                    # cfg_lines = list()
                    # cfg_path = os.path.join(
                    #     joern_parse_result_dir,
                    #     project, testcase_id, path)
                    # if os.path.exists(cfg_path) and os.path.isdir(cfg_path):
                    #     cfgs = extract_cfgs(cfg_path)
                    # else:
                    #     cfgs = []

                    # for cfg in cfgs:
                    #     for bb in cfg["nodes"]:
                    #         if line in bb:
                    #             cfg_lines.extend(bb)
                    # bug_info['cfg_lines'] = cfg_lines
                    bug_info_list.append(bug_info)
    with open('btp_classified_test.json', 'w') as f:
        json.dump(json_dict, f, indent=2)
        f.close()


def btp_unique():
    """
    @description  : 
                    The result of btp is deduplicated and calculated
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    with open('btp_classified_test.json', 'r') as f:
        json_dict = json.load(f)
        f.close()
    
    for bug_type in json_dict:
        bug_info_list = json_dict[bug_type]['bug_info']
        sample_count = 0
        BTP_sy_f1_bug = 0
        BTP_sy_precision_bug = 0
        BTP_sy_recall_bug = 0

        BTP_vd_f1_bug = 0
        BTP_vd_precision_bug = 0
        BTP_vd_recall_bug = 0

        BTP_method_f1_bug = 0
        BTP_method_precision_bug = 0
        BTP_method_recall_bug = 0
        # cfg_overlap = 0
        for bug_info in bug_info_list:
            sample_count += 1

            

            method_overlap = 0
            sy_overlap = 0
            vd_overlap = 0
            path = bug_info['path']
            trace_line = set(bug_info['trace_line'])
            
            bug_info['trace_line'] = list(trace_line)
            method_line = bug_info['method_line']
            sy_lines = set(bug_info['sy_lines'])
            vd_lines = set(bug_info['vd_lines'])
            # cfg_lines = set(bug_info['cfg_lines'])
            bug_info['sy_lines'] = list(sy_lines)
            bug_info['vd_lines'] = list(vd_lines)
            # bug_info['cfg_lines'] = list(cfg_lines)
            method_lines = method_line.get('lines', [])
            
            trace_line_count = len(trace_line)
            method_line_count = len(method_lines)
            sy_slice_line_count = len(sy_lines)
            vd_slice_line_count = len(vd_lines)
            # cfg_slice_line_count += len(cfg_lines)
            # method_overlap = method_overlap + len(trace_line) + len(method_lines)
            # sy_overlap = sy_overlap + len(trace_line) + len(sy_lines)
            # vd_overlap = vd_overlap + len(trace_line) + len(vd_lines)
            # cfg_overlap = cfg_overlap + len(trace_line) + len(cfg_lines)
            for l in trace_line:
                file_path = l.split(' ')[0]
                line = l.split(' ')[1]
                if file_path == path and int(line) in method_lines:
                    method_overlap += 1
                if file_path == path and int(line) in sy_lines:
                    sy_overlap += 1
                if file_path == path and int(line) in vd_lines:
                    vd_overlap += 1
                # if file_path == path and int(line) in cfg_lines:
                #     cfg_overlap += 1
            if method_line_count == 0:
                method_line_count = 0.0000001
            if sy_slice_line_count == 0:
                sy_slice_line_count = 0.0000001
            if vd_slice_line_count == 0:
                vd_slice_line_count = 0.0000001
            if trace_line_count == 0:
                trace_line_count = 0.0000001
            
            #method_matric
            BTP_method_precision = method_overlap / method_line_count
            BTP_method_recall = method_overlap / trace_line_count
            if BTP_method_precision != 0 or BTP_method_recall != 0:
                BTP_method_f1 = 2 * (BTP_method_precision * BTP_method_recall) / (
                    BTP_method_precision + BTP_method_recall) 
            else:
                BTP_method_f1 = 0
            BTP_method_precision_bug += BTP_method_precision
            BTP_method_recall_bug += BTP_method_recall
            BTP_method_f1_bug += BTP_method_f1
            bug_info['method_metric'] = dict()
            bug_info['method_metric'][
                'precision'] = BTP_method_precision
            bug_info['method_metric']['recall'] = BTP_method_recall
            bug_info['method_metric']['f1'] = BTP_method_f1

            #sy_matric
            BTP_sy_precision = sy_overlap / sy_slice_line_count
            BTP_sy_recall = sy_overlap / trace_line_count
            if BTP_sy_precision != 0 or BTP_sy_recall != 0:
                BTP_sy_f1 = 2 * (BTP_sy_precision * BTP_sy_recall) / (BTP_sy_precision +
                                                              BTP_sy_recall)
            else:
                BTP_sy_f1 = 0
            BTP_sy_precision_bug += BTP_sy_precision
            BTP_sy_recall_bug += BTP_sy_recall
            BTP_sy_f1_bug += BTP_sy_f1

            bug_info['sy_metric'] = dict()
            bug_info['sy_metric']['precision'] = BTP_sy_precision
            bug_info['sy_metric']['recall'] = BTP_sy_recall
            bug_info['sy_metric']['f1'] = BTP_sy_f1
            #vd_matric
            BTP_vd_precision = vd_overlap / vd_slice_line_count
            BTP_vd_recall = vd_overlap / trace_line_count
            if BTP_vd_precision != 0 or BTP_vd_recall != 0:
                BTP_vd_f1 = 2 * (BTP_vd_precision * BTP_vd_recall) / (BTP_vd_precision +
                                                               BTP_vd_recall)
            else:
                BTP_vd_f1 = 0   
            BTP_vd_precision_bug += BTP_vd_precision
            BTP_vd_recall_bug += BTP_vd_recall
            BTP_vd_f1_bug += BTP_vd_f1
            bug_info['vd_metric'] = dict()
            bug_info['vd_metric']['precision'] = BTP_vd_precision
            bug_info['vd_metric']['recall'] = BTP_vd_recall
            bug_info['vd_metric']['f1'] = BTP_vd_f1

        json_dict[bug_type]['method_metric'] = dict()
        json_dict[bug_type]['method_metric']['precision'] = BTP_method_precision_bug / sample_count
        json_dict[bug_type]['method_metric']['recall'] = BTP_method_recall_bug / sample_count
        json_dict[bug_type]['method_metric']['f1'] = BTP_method_f1_bug / sample_count

        json_dict[bug_type]['sy_metric'] = dict()
        json_dict[bug_type]['sy_metric']['precision'] = BTP_sy_precision_bug / sample_count
        json_dict[bug_type]['sy_metric']['recall'] = BTP_sy_recall_bug / sample_count
        json_dict[bug_type]['sy_metric']['f1'] = BTP_sy_f1_bug / sample_count

        json_dict[bug_type]['vd_metric'] = dict()
        json_dict[bug_type]['vd_metric']['precision'] = BTP_vd_precision_bug / sample_count
        json_dict[bug_type]['vd_metric']['recall'] = BTP_vd_recall_bug / sample_count
        json_dict[bug_type]['vd_metric']['f1'] = BTP_vd_f1_bug / sample_count
        

    with open('btp_unique_classified_test.json', 'w') as f:
        json.dump(json_dict, f, indent=2)
        f.close()


def btp_metric():
    """
    @description  : caculate btp_metric
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    
    with open('btp_unique_classified_test.json', 'r') as f:
        json_dict = json.load(f)
        f.close()
    
    sample_count = 0
    BTP_sy_f1_all = 0
    BTP_sy_precision_all = 0
    BTP_sy_recall_all = 0

    BTP_vd_f1_all = 0
    BTP_vd_precision_all = 0
    BTP_vd_recall_all = 0

    BTP_method_f1_all = 0
    BTP_method_precision_all = 0
    BTP_method_recall_all = 0
    btp_metric = dict()
    for bug_type in json_dict:
        bug_info_list = json_dict[bug_type]['bug_info']
        btp_metric[bug_type] = dict()
        btp_metric[bug_type]['method_metric'] = json_dict[bug_type]['method_metric']
        btp_metric[bug_type]['sy_metric'] = json_dict[bug_type]['sy_metric']
        btp_metric[bug_type]['vd_metric'] = json_dict[bug_type]['vd_metric']

        for bug_info in bug_info_list:
            sample_count += 1 
            BTP_sy_precision_all += bug_info['sy_metric']['precision']
            BTP_sy_recall_all += bug_info['sy_metric']['recall']
            BTP_sy_f1_all += bug_info['sy_metric']['f1']

            BTP_vd_precision_all += bug_info['vd_metric']['precision']
            BTP_vd_recall_all += bug_info['vd_metric']['recall']
            BTP_vd_f1_all += bug_info['vd_metric']['f1']

            BTP_method_precision_all += bug_info['method_metric']['precision']
            BTP_method_recall_all += bug_info['method_metric']['recall']
            BTP_method_f1_all += bug_info['method_metric']['f1']
    
    btp_metric['method_matric_all'] = dict()
    btp_metric['method_matric_all']['precision'] = BTP_method_precision_all / sample_count
    btp_metric['method_matric_all']['recall'] = BTP_method_recall_all / sample_count
    btp_metric['method_matric_all']['f1'] = BTP_method_f1_all / sample_count

    
    btp_metric['sy_matric_all'] = dict()
    btp_metric['sy_matric_all']['precision'] = BTP_sy_precision_all / sample_count
    btp_metric['sy_matric_all']['recall'] = BTP_sy_recall_all / sample_count
    btp_metric['sy_matric_all']['f1'] = BTP_sy_f1_all / sample_count
    
    btp_metric['vd_matric_all'] = dict()
    btp_metric['vd_matric_all']['precision'] = BTP_vd_precision_all / sample_count
    btp_metric['vd_matric_all']['recall'] = BTP_vd_recall_all / sample_count
    btp_metric['vd_matric_all']['f1'] = BTP_vd_f1_all / sample_count


    with open('btp_metric_classified_test.json', 'w') as f:
        json.dump(btp_metric, f, indent=2)
        f.close()
     

def plot_cwe_f1(vul_type: str):
    plt.rcParams['figure.figsize'] = (18, 12)
    recall = list()
    precision = list()
    f1 = list()
    labels = ["method", "vuldeepecker", "sysevr"]
    size = 3
    x = np.arange(size)
    total_width, n = 0.6, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    with open("btp_metric_classified.json", "r", encoding="utf-8") as f:
        btp_metrics = json.load(f)
    for key in ["method_matric", "vd_matric", "sy_matric"]:

        recall.append(btp_metrics[vul_type][key]["recall"])
        precision.append(btp_metrics[vul_type][key]["precision"])
        f1.append(btp_metrics[vul_type][key]["f1"])

    fig, ax = plt.subplots()
    ax.set_title(vul_type, fontsize=15)
    rects1 = ax.bar(x, precision, width=width, label='precision')
    rects2 = ax.bar(x + width,
                    recall,
                    width=width,
                    label='recal',
                    tick_label=labels)
    rects3 = ax.bar(x + 2 * width, f1, width=width, label='f1')
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    plt.xticks(x + width, labels, rotation=20, fontsize=15)
    plt.legend(loc='upper left', fontsize=12)
    out_dir = '../res/d2a/plot_png'
    plt.savefig(path.join(out_dir, vul_type + ".png"))
    plt.close()

def average_number_of_statement():
    with open('btp_unique_classified_test.json', 'r') as f:
        json_dict = json.load(f)
        f.close()
    sample_count = 0
    method_line_count = 0
    cdg_line_count = 0
    sevc_line_count = 0
    
    for bug_type in json_dict:
        bug_info = json_dict[bug_type]['bug_info']
        method_line_count += json_dict[bug_type]['method_line_count']
        cdg_line_count += json_dict[bug_type]['vd_slice_line_count']
        sevc_line_count += json_dict[bug_type]['sy_slice_line_count']
        sample_count += len(bug_info)
    print('sample_count: '+ str(sample_count))
    print('method_line_count: '+ str(method_line_count) + ' average_number_of_statement: '+str(method_line_count/sample_count))
    print('cdg_line_count: '+ str(cdg_line_count) + ' average_number_of_statement: '+str(cdg_line_count/sample_count))
    print('sevc_line_count: '+ str(sevc_line_count) + ' average_number_of_statement: '+str(sevc_line_count/sample_count))

def bug_triggering_path(bug_type):
    with open('btp_unique_classified_test.json', 'r') as f:
        json_dict = json.load(f)
        f.close()    

    trace_line_count = 0
    sample_count = 0
    for bug_type in json_dict:
        bug_info = json_dict[bug_type]['bug_info']
        trace_line_count += json_dict[bug_type]['trace_line_count']    
        sample_count += len(bug_info)
    return sample_count, trace_line_count

if __name__ == '__main__':
   pass



