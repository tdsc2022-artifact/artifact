import os
from slicing import get_data, get_data_d2a
import sys
sys.path.append("..")
from utils.xml_parser import getCodeIDtoPathDict_osp, getCodeIDtoPathDict_d2a
import xml.etree.ElementTree as ET
import json

def joern_parse_d2a(data_dir):
    """
    @description  : use joern to parse c/cpp
    ---------
    @param  : data_dir: c/cpp dir
    -------
    @Returns  :
    -------
    """
    
    
    workspace = 'joern'
    os.chdir(workspace)
    for project in os.listdir(data_dir):
        cmd = './joern-parse d2a_resources_btp/'+project+' '+os.path.join(data_dir, project)
        print('CMD: '+cmd)
        os.system(cmd)



def slicing_d2a(source_root_dir):
    """
    @description  : slice D2A from the result from joern parse
    ---------
    @param  : sourece_root_dir: the root dir of all project source code
    -------
    @Returns  :
    -------
    """
    
    
    root = 'joern/d2a_resources_btp'
    out_dir = 'joern/d2a_slice_btp'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for project in os.listdir(root):
        if not project == 'libav':
            continue
        xmlPath = os.path.join(source_root_dir, project, 'source-code','manifest_btp.xml')
        tree = ET.ElementTree(file=xmlPath)
        testcases = tree.findall("testcase")
        codeIDtoPath = getCodeIDtoPathDict_d2a(testcases, source_root_dir)
        # if project != 'redis':
        #     continue
        project_dir = os.path.join(out_dir, project)
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
        for testcase in testcases:
            testcaseID = testcase.attrib["id"]
            data = []
            testcase_dir = os.path.join(root, project)
            for file_path in codeIDtoPath[testcaseID].keys():
                data_dir = os.path.join(testcase_dir, testcaseID, file_path)
                if os.path.isdir(data_dir):
                    print(data_dir)
                    data.extend(get_data_d2a(data_dir))
            with open(os.path.join(project_dir, testcaseID+'.json'),'w') as f:
                json.dump(data, f, indent=2)
                f.close()
                    


def generate_cdg_d2a(project, source_root_dir, cdg_vg_out_path, cdg_sy_out_path):
    
    """
    @description  : generate Code gadget and SeVC samples
    ---------
    @param  :project: project name
             source_root_dir: the root dir of all project source code
             cdg_vg_out_path: the output dir of samples of VulDeepecker
             cdg_sy_out_path: the output dir of samples of SySeVR
    -------
    @Returns  :
    -------
    """
    
    xmlPath = os.path.join(source_root_dir, project, 'source-code', "manifest_bpt.xml")
    tree = ET.ElementTree(file=xmlPath)
    testcases = tree.findall("testcase")
    print("start - generating codeIDtoPath dict...")
    codeIDtoPath = getCodeIDtoPathDict_d2a(
        testcases, source_root_dir)  # {testcaseid:{filePath:set(vul lines)}}
    
    if not os.path.exists(cdg_vg_out_path):
        os.makedirs(cdg_vg_out_path)
    
    if not os.path.exists(cdg_sy_out_path):
        os.makedirs(cdg_sy_out_path)
    


    for testcase in testcases:
        testcaseID = testcase.attrib["id"]
        filePathList = list(codeIDtoPath[testcaseID].keys())
       
        slice_root_dir = os.path.join('joern/d2a_slice', project)
        
        slice_dir = os.path.join(slice_root_dir, testcaseID)
        if os.path.exists(slice_dir+'.json'):
            with open(slice_dir+'.json', 'r') as f:
                slices = json.load(f)
                f.close()
        else:
                slices = []
        for sl in slices:
            file_path = sl['file_path']
            file_path_rear = '/'.join(file_path.split('/')[11:])
            for file in filePathList:
                if file_path_rear == file:
                    key = file
                    break
            file_path = '/'+'/'.join(file_path.split('/')[3:])
            
            with open(file_path,
                          "r",
                          encoding="utf-8",
                          errors="ignore") as f:
                file_content = f.readlines()
                f.close()
            for s in sl:
                if s.endswith('_vd'):
                    cdgs = sl[s]
                    for cdg in cdgs:
                        if (len(cdg) < 3):
                            continue
                        info = str(testcaseID) + " " + file_path + " " + "\n"
                        content = ""
                        target = -1  
                        for line in cdg:
                            content = content + file_content[int(line) -1].strip() + "\n"
                            if target == 0:
                                continue
                            if str(line) in codeIDtoPath[testcaseID][
                                key]['SafeLine'].keys():
                                key_line = str(line)
                                target = 0
                        if target != 0:
                            continue
                        for bug_type in codeIDtoPath[testcaseID][key]['SafeLine'][key_line]:
                            cdg_vg_out_dir = os.path.join(cdg_vg_out_path, bug_type)
                            if not os.path.exists(cdg_vg_out_dir):
                                os.mkdir(cdg_vg_out_dir)
                            with open(cdg_vg_out_dir+'/all.txt',
                                     "a",
                                    encoding="utf-8",
                                    errors="ignore") as f:
                                f.write(info + content + str(target) + "\n" +
                                        "---------------------------------" + "\n")
                                f.close()   
                elif s.endswith('_sy'):
                    cdgs = sl[s]
                    for cdg in cdgs:
                        if (len(cdg) < 3):
                            continue
                        info = str(testcaseID) + " " + file_path + " " + "\n"
                        content = ""
                        target = -1  
                        for line in cdg:
                            content = content + file_content[int(line) -1].strip() + "\n"
                            if target == 0:
                                continue
                            if str(line) in codeIDtoPath[testcaseID][
                                key]['SafeLine'].keys():
                                key_line = str(line) 
                                target = 0
                        if target != 0:
                            continue
                        for bug_type in codeIDtoPath[testcaseID][key]['SafeLine'][key_line]:
                            cdg_sy_out_dir = os.path.join(cdg_sy_out_path, bug_type)
                            if not os.path.exists(cdg_sy_out_dir):
                                os.mkdir(cdg_sy_out_dir)
                            with open(cdg_sy_out_dir+'/all.txt',
                                     "a",
                                    encoding="utf-8",
                                    errors="ignore") as f:
                                f.write(info + content + str(target) + "\n" +
                                        "---------------------------------" + "\n")
                                f.close()

        #vul
        
        for sl in slices:
            file_path = sl['file_path']
            file_path_rear = '/'.join(file_path.split('/')[11:])
            for file in filePathList:
                if file_path_rear == file:
                    key = file
                    break
            file_path = '/' + '/'.join(file_path.split('/')[3:])
            with open(file_path,
                          "r",
                          encoding="utf-8",
                          errors="ignore") as f:
                file_content = f.readlines()
                f.close()
            for s in sl:
                if s.endswith('_vd'):
                    cdgs = sl[s]
                    for cdg in cdgs:
                        if (len(cdg) < 3):
                            continue
                        info = str(testcaseID) + " " + file_path + " " + "\n"
                        content = ""
                        target = -1  
                        for line in cdg:
                            content = content + file_content[int(line) -1].strip() + "\n"
                            if target == 1:
                                continue
                            if str(line) in codeIDtoPath[testcaseID][
                                key]['VulLine'].keys():
                                key_line = str(line) 
                                target = 1
                        if target != 1:
                            continue
                        for bug_type in codeIDtoPath[testcaseID][key]['VulLine'][key_line]:
                            cdg_vg_out_dir = os.path.join(cdg_vg_out_path, bug_type)
                            if not os.path.exists(cdg_vg_out_dir):
                                os.mkdir(cdg_vg_out_dir)
                            with open(cdg_vg_out_dir + '/all.txt',
                                     "a",
                                    encoding="utf-8",
                                    errors="ignore") as f:
                                f.write(info + content + str(target) + "\n" +
                                        "---------------------------------" + "\n")
                                f.close()   
                elif s.endswith('_sy'):
                    cdgs = sl[s]
                    for cdg in cdgs:
                        if (len(cdg) < 3):
                            continue
                        info = str(testcaseID) + " " + file_path + " " + "\n"
                        content = ""
                        target = -1  
                        for line in cdg:
                            content = content + file_content[int(line) -1].strip() + "\n"
                            if target == 1:
                                continue
                            if str(line) in codeIDtoPath[testcaseID][
                                key]['VulLine'].keys():
                                key_line = str(line) 
                                target = 1
                        if target != 1:
                            continue
                        for bug_type in codeIDtoPath[testcaseID][key]['VulLine'][key_line]:
                            cdg_sy_out_dir = os.path.join(cdg_sy_out_path, bug_type)
                            if not os.path.exists(cdg_sy_out_dir):
                                os.mkdir(cdg_sy_out_dir)
                            with open(cdg_sy_out_dir + '/all.txt',
                                     "a",
                                    encoding="utf-8",
                                    errors="ignore") as f:
                                f.write(info + content + str(target) + "\n" +
                                        "---------------------------------" + "\n")
                                f.close()  

if __name__ == "__main__":
    # root = 'joern/d2a_resources'
    # for project in os.listdir(root):
    #     # if project == 'httpd':
    #     #     continue
    #     try:
    #         generate_cdg_d2a(project)
    #     except Exception as e:
    #         with open('generate_cdg_d2a_error.log', 'a') as f:
    #             f.write(project+'\n')
    #             f.write(str(e)+'\n')
    #             f.close()
    # generate_cdg_d2a('httpd')
    # joern_parse_d2a()
    slicing_d2a()