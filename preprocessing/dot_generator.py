'''
generate .bc and .dot automatically

@author : jumormt
@version : 1.0
'''
import os
from os.path import join
from omegaconf import DictConfig
import xml.etree.ElementTree as ET
from typing import Union
from utils import git_checkout
import shutil


def generate_bc_osp(data_root_dir: str, project: str):

    xml_path = join(data_root_dir, project, 'source-code', 'manifest.xml')

    src_root = join(data_root_dir, project, 'source-code', project)

    repo_dir = src_root

    bc_dir = join(data_root_dir, project, 'bc')
    if not os.path.exists(bc_dir):
        os.mkdir(bc_dir)

    # if(len(os.listdir(bc_dir)) != 0 ):
    #     print(f"{project} bc already exists!")
    #     return

    tree = ET.ElementTree(file=xml_path)
    data = tree.findall("testcase")
    testcaseList = list()
    coutn = 0
    for testcase in data:
        #checkout

        testcaseList.append(testcase)
        testcaseid = testcase.attrib["id"]
        testcase_dir = join(bc_dir, testcaseid)
        if not os.path.exists(testcase_dir):
            os.mkdir(testcase_dir)

        if len(
                os.listdir(testcase_dir)
        ) > 0 and 'compile_error_id.txt' not in os.listdir(testcase_dir):
            continue
        #git checkout

        git_r = git_checkout.checkout_to(repo_dir, testcaseid, project)
        # if testcaseid in doneIDs:
        #     continue

        #编译
        print('start compile ...')
        os.chdir(src_root)
        CMD = "CC=wllvm CXX=wllvm++ ./config"
        if os.system(CMD) != 0:
            with open(join(testcase_dir, 'compile_error_id.txt'), 'a+') as f:
                f.write(testcaseid + ' configure error!' '\n')
                f.close()
        CMD = "CC=wllvm CXX=wllvm++ make"
        if os.system(CMD) != 0:
            with open(join(testcase_dir, 'compile_error_id.txt'), 'a+') as f:
                f.write(testcaseid + ' make error'+'\n')
                f.close()
        print('compile end')
        #编译

        #提取BC文件
        file_x = set()
        x_dir = join(src_root, 'apps')
        # x_dir = src_root
        for file in os.listdir(x_dir):
            file_f = os.path.join(x_dir, file)
            if os.access(file_f, os.X_OK) and os.path.isfile(
                    file_f) and file.find('.') == -1:
                file_x.add(file)
        for file in file_x:
            print('extract bc id :{} file : {} ....'.format(testcaseid, file))

            CMD = "extract-bc {} -o {}".format(
                join(x_dir, file), join(testcase_dir, file + '.bc'))
            os.system(CMD)
            print('CMD: ' + CMD)
        print("TESTCASEID {} DONE!".format(testcaseid))
        # print()
        #checkout_back
        git_checkout.checkout_back(git_r, project)

    print("ALL WORK DONE! TOTAL {} TESTCASES.".format(coutn))


def generate_bc(data_root_dir: str, cweid: str):
    '''
    TODO: open-source projects
    generate llvm bitcode
    :param data_root: /home/chengxiao/dataset/vul/CWE
    :param cweid: e.g., CWE119
    :return:
    '''
    xml_path = join(data_root_dir, cweid, "sard", "source-code",
                    "manifest.xml")
    src_root = join(data_root_dir, cweid, "sard", "source-code")

    bc_dir = join(data_root_dir, cweid, "sard", "bc")

    bc_tmp_dir = join(data_root_dir, cweid, "sard", "bc_tmp")
    if (len(os.listdir(bc_dir)) != 0):
        print(f"{cweid} bc already exists!")
        return

    tree = ET.ElementTree(file=xml_path)
    data = tree.findall("testcase")
    testcaseList = list()
    coutn = 0

    for testcase in data:  # for each testcase
        testcaseList.append(testcase)
        testcaseid = testcase.attrib["id"]
        # if testcaseid in doneIDs:
        #     continue

        files = testcase.findall("file")
        file_f_set = set()  # may contain more than one file(a. b. c. etc.)
        includepathStr = ""
        includepathVisited = set()
        for file in files:

            path = file.attrib["path"]
            flaws = file.find("flaw")
            mixed = file.find("mixed")
            fix = file.find("fix")
            if (flaws is not None or mixed is not None
                    or fix is not None):  # get file to analyze
                file_f = join(src_root, path)
                print("FILE:  ", file_f)
                file_f_set.add(file_f)
            else:  # get include dir
                idx = path.rfind("/")
                includepath = join(src_root, path[:idx])
                if includepath not in includepathVisited:
                    includepathVisited.add(includepath)
                    includepathStr = includepathStr + "-I" + includepath + " "
                    print("INCLUDEPATH: ", includepath)

        if (len(file_f_set) == 1):  # for single file
            cmd = "clang -c -emit-llvm -g {file_f} {include} -o {out}.bc".format(
                file_f=list(file_f_set)[0],
                include=includepathStr,
                out=join(bc_dir, str(testcaseid)))
            print("CMD: ", cmd)
            os.system(cmd)
        else:
            file_f_set = list(file_f_set)
            tmpfiles = ""
            for i in range(len(file_f_set)):
                fil = file_f_set[i]
                cmd = "clang -c -emit-llvm -g {file_f} {include} -o {out}.bc".format(
                    file_f=fil,
                    include=includepathStr,
                    out=join(bc_tmp_dir,
                             str(testcaseid) + "_" + str(i)))

                tmpfile = join(bc_tmp_dir,
                               str(testcaseid) + "_" + str(i) + ".bc")
                tmpfiles = tmpfiles + tmpfile + " "
                print("CMD: ", cmd)
                os.system(cmd)
            cmd = "llvm-link " + tmpfiles + "-o {}.bc".format(
                join(bc_dir, str(testcaseid)))
            print("CMD: ", cmd)
            os.system(cmd)

        coutn = coutn + 1
        print("TESTCASEID {} DONE!".format(testcaseid))
        # print()

    print("ALL WORK DONE! TOTAL {} TESTCASES.".format(coutn))

    # print()


def generate_VFG_osp(data_root_dir: str, project: str):
    bc_dir = join(data_root_dir, project, 'bc')
    dot_dir = join(data_root_dir, project, 'dot')
    if not os.path.exists(dot_dir):
        os.mkdir(dot_dir)
    for testcase in os.listdir(bc_dir):
        for bc_file in os.listdir(join(bc_dir, testcase)):
            if (os.path.splitext(bc_file)[-1] != ".bc"):
                continue

            fullpath = join(bc_dir, testcase, bc_file)
            print("{} - processing..".format(fullpath))
            outputFileDir = join(dot_dir, testcase,
                                 os.path.splitext(bc_file)[0])
            if (not os.path.exists(outputFileDir)):
                os.makedirs(outputFileDir)
            elif (len(os.listdir(outputFileDir)) == 10):
                print(f"{outputFileDir} contains dots!")
                continue

            os.chdir(outputFileDir)

            cmd0 = "wpa -ander -svfg -dump-svfg {}".format(
                fullpath)  # Dump Value-Flow Graph
            cmd1 = "wpa -ander -dump-pag {}".format(
                fullpath)  # Dump PAG (program assignment graph)
            cmd2 = "wpa -ander -dump-consG {}".format(
                fullpath)  # Dump Constraint Graph
            cmd3 = "wpa -ander -dump-callgraph {}".format(
                fullpath)  # Dump Callgraph
            cmd4 = "wpa -ander -svfg -dump-mssa {}".format(
                fullpath)  # Dump Memory SSA
            cmd5 = "wpa -type -genicfg -dump-icfg {}".format(
                fullpath)  # Dump Memory SSA
            os.system(cmd0)
            os.system(cmd1)
            os.system(cmd2)
            os.system(cmd3)
            os.system(cmd4)
            os.system(cmd5)

            print("{} - end!".format(bc_file))


def generate_VFG(data_root_dir: str, cweid: str):
    '''
    TODO: open-source projects
    generate graphs using SVF
    :param inputDir:
    :param outputDir:
    :return:
    '''
    bc_dir = join(data_root_dir, cweid, "sard", "bc")
    dot_dir = join(data_root_dir, cweid, "sard", "dot")
    for dirpath, dirnames, filenames in os.walk(bc_dir):

        for file in filenames:  # for each bc
            if (os.path.splitext(file)[-1] != ".bc"):
                continue
            print("{} - processing..".format(file))
            fullpath = join(dirpath, file)
            outputFileDir = join(dot_dir, os.path.splitext(file)[0])
            if (not os.path.exists(outputFileDir)):
                os.mkdir(outputFileDir)
            elif (len(os.listdir(outputFileDir)) == 10):
                print(f"{outputFileDir} contains dots!")
                continue

            os.chdir(outputFileDir)

            cmd0 = "wpa -ander -svfg -dump-svfg {}".format(
                fullpath)  # Dump Value-Flow Graph
            cmd1 = "wpa -ander -dump-pag {}".format(
                fullpath)  # Dump PAG (program assignment graph)
            cmd2 = "wpa -ander -dump-consG {}".format(
                fullpath)  # Dump Constraint Graph
            cmd3 = "wpa -ander -dump-callgraph {}".format(
                fullpath)  # Dump Callgraph
            cmd4 = "wpa -ander -svfg -dump-mssa {}".format(
                fullpath)  # Dump Memory SSA
            cmd5 = "wpa -type -genicfg -dump-icfg {}".format(
                fullpath)  # Dump Memory SSA
            os.system(cmd0)
            os.system(cmd1)
            os.system(cmd2)
            os.system(cmd3)
            os.system(cmd4)
            os.system(cmd5)

            print("{} - end!".format(file))


def generate_bc_VFG_osp(_config: DictConfig, project: Union[str, None]):
    data_root_dir = _config.raw_data_folder
    cve_root_dir = os.path.join(data_root_dir, 'CVE')
    if project:
        generate_bc_osp(cve_root_dir, project)
        generate_VFG_osp(cve_root_dir, project)
    else:
        projects = os.listdir(cve_root_dir)
        for project in projects:

            generate_bc_osp(cve_root_dir, project)
            generate_VFG_osp(cve_root_dir, project)


def generate_bc_VFG(_config: DictConfig, cweid: Union[str, None]):

    data_root_dir = _config.raw_data_folder
    cwe_root_dir = join(data_root_dir, 'CWE')
    if cweid:
        generate_bc(cwe_root_dir, cweid)
        generate_VFG(cwe_root_dir, cweid)
    else:
        cweids = os.listdir(cwe_root_dir)
        for cweid in cweids:
            generate_bc(cwe_root_dir, cweid)
            generate_VFG(cwe_root_dir, cweid)
