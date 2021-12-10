import json
import re
import os
import shutil
from git import Repo, Git
from pydriller import RepositoryMining
from pydriller import GitRepository
import time
import sys

from utils.git_checkout import checkout_back, checkout_to
import xml.etree.ElementTree as ET


def classify_by_cweid(root_dir, out_dir):
    """

    Args:
        root_dir: commit_info 目录 ../../commit_info/
        out_dir: 输出json文件的目录 ../../output/

    Returns:

    """

    all_file = {}

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            all_file[dir] = list()
            for file in os.listdir(os.path.join(root_dir, dir)):
                result = re.findall('detail.json', file)
                if len(result) > 0:
                    pass
                else:
                    all_file[dir].append(os.path.join(root_dir, dir, file))
    result_data = dict()
    cwe_set = set()
    for dir in all_file:
        source = dir
        for file in all_file[dir]:
            with open(file, 'r', encoding='utf8') as f:
                json_data = json.load(f)
                product = json_data['product']
                commit_list = json_data[product]
                for commit in commit_list:
                    for key in commit:
                        value = commit[key]
                        cve_info = value['cve_info']
                        for cve in cve_info:
                            cwe_info = cve_info[cve]
                            for key in cwe_info:
                                cwes = cwe_info[key]
                                for cwe_id in cwes:
                                    if len(
                                            re.findall('CWE-[0-9]+',
                                                       str(cwe_id), re.I)) > 0:
                                        block = dict()
                                        block['commit_info'] = commit
                                        block['product'] = product
                                        block['source'] = source
                                        if cwe_id in cwe_set:
                                            commits = result_data[cwe_id][
                                                'commits']
                                            if commit in commits:
                                                pass
                                            else:
                                                commits.append(block)
                                                result_data[cwe_id][
                                                    'commits'] = commits
                                        else:
                                            result_data[cwe_id] = dict()
                                            commits = list()
                                            commits.append(block)
                                            result_data[cwe_id][
                                                'commits'] = commits
                                            cwe_set.add(cwe_id)

    for cwe_id in cwe_set:
        commits = result_data[cwe_id]['commits']
        result_data[cwe_id]['commits_count'] = len(commits)

    with open(out_dir + 'final.json', 'wb') as f:
        json_str = json.dumps(result_data, indent=2)
        f.write(json_str.encode())
        f.close()


def classify_by_project(root_dir, out_dir):
    all_file = {}

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            all_file[dir] = list()
            for file in os.listdir(os.path.join(root_dir, dir)):
                result = re.findall('detail.json', file)
                if len(result) > 0:
                    pass
                else:
                    all_file[dir].append(os.path.join(root_dir, dir, file))
    result_data = dict()

    #遍历200文件
    for dir in all_file:
        file_list = all_file[dir]
        for file in file_list:
            with open(file, 'r', encoding='utf8') as f:
                json_data = json.load(f)
                product = json_data['product']
                commit_list = json_data[product]
                c_list = list()
                for commit_info in commit_list:
                    commit_data = {}
                    for commit_id in commit_info:
                        commit = commit_info[commit_id]
                        cve_info = commit['cve_info']
                        commit_data['commit_id'] = commit_id
                        commit_data['cve_info'] = cve_info
                        c_list.append(commit_data)
                if len(c_list) > 0:
                    result_data[product] = {}
                    result_data[product]['info'] = c_list
                    result_data[product]['source_code'] = dir
    with open(out_dir + 'final.json', 'wb') as f:
        json_str = json.dumps(result_data, indent=2)
        f.write(json_str.encode())
        f.close()


def load_final_json(json_dir):
    """

    Args:
        json_dir: final.json 的目录

    Returns:json_dic

    """
    json_dic = {}
    with open(json_dir, 'r') as f:
        json_dir = json.load(f)
        json_dic = dict(json_dir)
        f.close()
    return json_dic


def parse_json_old(json_dic, dataset_dir, repo_dir):
    """

    Args:
        json_dic: final_json字典
        dataset_dir: 重组数据集存放的目录
        repo_dir: project repository dir

    Returns:

    """

    print('start parse json to xml')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for cwe_id in json_dic:
        c_count = 0
        cpp_count = 0
        s_type = 'Source Code'
        status = 'Accepted'
        cwe_dir = os.path.join(dataset_dir, cwe_id.replace('-', ''),
                               'cve')  # dataset/CWE119/cve/
        if not os.path.exists(cwe_dir):
            os.makedirs(cwe_dir)
        source_code_dir = os.path.join(
            cwe_dir, 'source-code')  # dataset/CWE119/cve/source-code
        if not os.path.exists(source_code_dir):
            os.makedirs(source_code_dir)
        cwe_info = json_dic[cwe_id]
        commits_list = cwe_info['commits']
        for commit in commits_list:
            root = ET.Element('container')
            commit_info = commit['commit_info']
            product = commit['product']
            source = commit['source']
            language = source
            if source == 'c':
                dir_str = ''.join(cwe_id[4:]) + '-' + '%03d' % c_count + '-c'
                source_code = 'c'
                c_count = c_count + 1
            else:
                source_code = 'cpp'
                dir_str = ''.join(
                    cwe_id[4:]) + '-' + '%03d' % cpp_count + '-cpp'
                cpp_count = cpp_count + 1
            code_dir = os.path.join(source_code_dir, dir_str)
            testcase_dir = os.path.join(code_dir, 'testcase')
            if not os.path.exists(testcase_dir):
                os.makedirs(testcase_dir)
            for commit_id in commit_info:
                file_info = commit_info[commit_id]
                cve_info = file_info['cve_info']
                commit_msg = file_info['commit_msg']
                description = commit_msg

                try:
                    for detail in RepositoryMining(
                            os.path.join(repo_dir, source, product),
                            single=commit_id).traverse_commits():
                        submissionDate = detail.committer_date.strftime(
                            '%Y-%m-%d')
                        author = detail.author
                        # 创建testcase 节点
                        testcase = ET.Element('testcase')
                        testcaseId = product + '_' + commit_id
                        testcase.set('id', testcaseId)
                        testcase.set('type', s_type)
                        testcase.set('status', status)
                        testcase.set('submissionDate', submissionDate)
                        testcase.set('language', source)
                        testcase.set('author', author.name)
                        # 创建testcase 节点

                        # 创建description节点
                        description = ET.Element('description')
                        description.text = detail.msg
                        # 创建description节点
                        for file in detail.modifications:
                            name = file.filename
                            new_path = file.new_path
                            old_path = file.old_path
                            diff_parsed = file.diff_parsed
                            del_lines = [i[0] for i in diff_parsed['deleted']]
                            code_before = file.source_code_before
                            code_after = file.source_code
                            new_name = 'new_' + name
                            old_name = 'old_' + name
                            code_dir_new = os.path.join(testcase_dir, new_name)
                            code_dir_old = os.path.join(testcase_dir, old_name)
                            print('writing source code')
                            with open(code_dir_new, 'wb') as f:
                                f.write(str(code_after).encode())
                                f.close()
                            print('write successful')
                            # 创建new_file节点
                            new_file_node = ET.Element('file')
                            new_file_node.set('path', new_name)
                            new_file_node.set('language', source)
                            # 创建new_file节点
                            print('writing source code')
                            with open(code_dir_old, 'wb') as f:
                                f.write(str(code_before).encode())
                                f.close()
                            print('write successful')
                            # 创建old_file节点
                            old_file_node = ET.Element('file')
                            old_file_node.set('path', old_name)
                            old_file_node.set('language', source)
                            if len(del_lines) > 0:
                                for line in del_lines:
                                    flaw = ET.Element('flaw')
                                    flaw.set('line', str(line))
                                    old_file_node.append(flaw)
                            # 创建old_file节点

                            # testcase 添加 file
                            testcase.append(description)
                            testcase.append(new_file_node)
                            testcase.append(old_file_node)
                            # container 添加 testcase
                            root.append(testcase)
                except Exception as e:
                    with open('./log.txt', 'a+') as f:
                        f.write(str(e) + '\n')
                        f.close()
                # 写入为manifest.xml
                print('writing manifest.xml')
                prettyXml(root, '\t', '\n')  # 执行美化方法
                # ET.dump(root)  # 显示出美化后的XML内容
                tree = ET.ElementTree(root)
                man_dir = os.path.join(code_dir, 'manifest.xml')
                tree.write(man_dir, 'utf-8', True)
                print('write successful!')
    print('parse successful!')


def prettyXml(element, indent, newline, level=0):
    """
    用于缩进xml输出
    Args:
        element: elemnt为传进来的Elment类
        indent: 参数indent用于缩进
        newline: newline用于换行
        level:

    Returns:

    """
    if element:  # 判断element是否有子元素
        if element.text == None or element.text.isspace(
        ):  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip(
            ) + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
        # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (
                len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def parse_json(json_dic, dataset_dir, repo_dir):

    s_type = 'Source Code'
    status = 'Accepted'
    print('start parse ...')
    for project in json_dic:
        # if not project == 'redis':
        #     continue
        source_code_dir = os.path.join(dataset_dir, project, 'source-code')
        if not os.path.exists(source_code_dir):
            os.makedirs(source_code_dir)
        product_info = json_dic[project]
        info_list = product_info['info']
        source_code = product_info['source_code']
        if source_code == 'c':
            source = 'c'
        else:
            source = 'cpp'

        source_dir = os.path.join(repo_dir, source_code, project)  # 工程源目录

        target_dir = os.path.join(source_code_dir, project)  # 工程现目录

        #判断是否已经建立
        # if os.path.exists(os.path.join(source_code_dir,'manifest.xml')):
        #     continue
        # else:
        if os.path.exists(target_dir):
            # shutil.rmtree(target_dir)
            pass
        else:
            print('start copy project files ...')
            shutil.copytree(source_dir, target_dir)
            print('copy successful ！！')
        container = ET.Element('container')

        repo = Repo(target_dir)
        r = Git(target_dir)

        for info in info_list:
            #遍历每个commit
            commit_id = info['commit_id']

            cve_info = info['cve_info']
            cwe_list = re.findall('cwe-[0-9]+', str(cve_info), re.I)
            try:
                for detail in RepositoryMining(
                        target_dir, single=commit_id).traverse_commits():
                    submissionDate = detail.committer_date.strftime('%Y-%m-%d')
                    author = detail.author
                    # 创建testcase 节点
                    testcase = ET.Element('testcase')
                    testcaseId = commit_id
                    testcase.set('id', testcaseId)
                    testcase.set('type', s_type)
                    testcase.set('status', status)
                    testcase.set('submissionDate', submissionDate)
                    testcase.set('language', source)
                    testcase.set('author', author.name)
                    # 创建testcase 节点

                    # 创建description节点
                    description = ET.Element('description')
                    description.text = detail.msg
                    testcase.append(description)
                    # 创建description节点

                    # 创建CVE节点

                    for cve_id in cve_info:
                        cve = ET.Element('CVE')
                        cve.set('id', cve_id)
                        testcase.append(cve)
                #创建CVE节点

                #创建CWE节点
                    for cwe in cwe_list:
                        cwe_ = ET.Element('CWE')
                        cwe_.set('id', cwe)
                        testcase.append(cwe_)
                # 创建CWE节点
                # 创建file节点
                    flaw_file_new_path = [
                        file.new_path.replace('\\', '/')
                        for file in detail.modifications if file.new_path
                    ]
                    flaw_file_old_path = [
                        file.old_path.replace('\\', '/')
                        for file in detail.modifications if file.old_path
                    ]
                    file_set = set(flaw_file_new_path + flaw_file_old_path)

                    #遍历与commit相关的文件
                    for file in detail.modifications:
                        diff_parsed = file.diff_parsed
                        del_lines = [i[0] for i in diff_parsed['deleted']]
                        add_lines = [i[0] for i in diff_parsed['added']]
                        if file.new_path:
                            new_path = file.new_path.replace('\\', '/')
                            # 创建new_file节点
                            new_file_node = ET.Element('file')
                            new_file_node.set('path', new_path)
                            new_file_node.set('language', source)
                            if len(del_lines) > 0:
                                for line in del_lines:
                                    flaw = ET.Element('flaw')
                                    flaw.set('type', str(1))
                                    flaw.set('line', str(line))
                                    new_file_node.append(flaw)
                            testcase.append(new_file_node)
                        # 创建new_file节点
                        if file.old_path:
                            old_path = file.old_path.replace('\\', '/')
                            
                            
                            # 创建old_file节点
                            old_file_node = ET.Element('file')
                            old_file_node.set('path', old_path)
                            old_file_node.set('language', source)
                            if len(add_lines) > 0:
                                for line in add_lines:
                                    flaw = ET.Element('flaw')
                                    flaw.set('type',str(0))
                                    flaw.set('line', str(line))
                                    old_file_node.append(flaw)
                            testcase.append(old_file_node)
                        # 创建old_file节点

                    # testcase 添加 file

                    #遍历与commit相关的文件

                    #遍历与commit不相关的文件
                    past_branch_name = project
                    repo.create_head(past_branch_name, commit_id)
                    try:
                        r.execute('git checkout ' + past_branch_name + ' -f',
                                  shell=True)
                    except:
                        r.execute('git branch -D ' + past_branch_name,
                                  shell=True)
                        r.execute('git checkout ' + past_branch_name + ' -f',
                                  shell=True)
                    not_flaw_files = [
                        os.path.relpath(file, target_dir).replace('\\', '/')
                        for file in GitRepository(target_dir).files()
                    ]
                    print(len(not_flaw_files), project)
                    try:
                        r.execute('git checkout master -f', shell=True)
                    except:
                        r.execute('git checkout unstable -f', shell=True)
                    r.execute('git branch -D ' + past_branch_name, shell=True)

                    # get the list of files present in the repo at the current commit
                    print('write file ....')
                    for file in not_flaw_files:
                        if file not in file_set:
                            if (
                                    not file.endswith(".Cpp")
                                    and not file.endswith(".c++")
                                    and not file.endswith(".C")
                                    and not file.endswith(".cxx")
                                    and not file.endswith(".c")
                                    and not file.endswith(".cpp")
                                    and not file.endswith(".cc")
                                    and not file.endswith(".h")
                            ):  # filter files that do not contain source code
                                continue
                            file_node = ET.Element('file')
                            file_node.set('path', file)
                            file_node.set('language', source)
                            testcase.append(file_node)
                #遍历与commit不相关的文件
                # container 添加 testcase
                    container.append(testcase)
            except Exception as e:
                # with open('log.txt','a+') as f:
                #     f.write(target_dir +'\n'+commit_id+'\n'+
                #     str(e)+'\n'+str(time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
                print(e)
        # 写入为manifest.xml
        print('writing ' + project + ' manifest.xml')
        prettyXml(container, '\t', '\n')  # 执行美化方法
        # ET.dump(root)  # 显示出美化后的XML内容
        tree = ET.ElementTree(container)
        man_dir = os.path.join(source_code_dir, 'manifest.xml')
        tree.write(man_dir, 'utf-8', True)
        print('write successful!')
    print('parse successful!')


if __name__ == '__main__':
   pass
