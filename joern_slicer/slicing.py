#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from os.path import join, isdir
import csv

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
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number


def create_adjacency_list(line_numbers,
                          node_id_to_line_numbers,
                          edges,
                          data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  #edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys(
            ) or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS':  #Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES':  # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list


def create_forward_slice(adjacency_list, line_no):
    sliced_lines = set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    sliced_lines = sorted(sliced_lines)
    return sliced_lines


def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    data_graph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])

        data_graph[ln] = set()
        data_graph[ln] = data_graph[ln].union(adjacency_list[ln][1])
    return cgraph, data_graph


def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph
    pass


def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)

def get_data(root):
    
    sensi_api_path = "../resources/sensiAPI.txt"
    cpg_list = [join(root, fl) for fl in os.listdir(root) if isdir(join(root, fl))]


    with open(sensi_api_path, "r", encoding="utf-8") as f:
        sensi_api_set = set([api.strip() for api in f.read().split(",")])


    all_data = list()
    for cpg in cpg_list:
        nodes_path = join(cpg, "nodes.csv")
        edges_path = join(cpg, "edges.csv")
        with open(nodes_path, "r") as f:
            nodes = [node for node in csv.DictReader(f, delimiter='\t')]
        call_lines = set()
        array_lines = set()
        ptr_lines = set()
        arithmatic_lines = set()
        if len(nodes) == 0:
            continue
        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            if ntype == 'CallExpression':
                function_name = nodes[node_idx + 1]['code']
                if function_name is None or function_name.strip() == '':
                    continue
                if function_name.strip() in sensi_api_set:
                    line_no = extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            elif ntype == 'ArrayIndexing':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            elif ntype == 'PtrMemberAccess':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            elif node['operator'].strip() in ['+', '-', '*', '/']:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)


        nodes = read_csv(nodes_path)
        edges = read_csv(edges_path)
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges,
                                               False)

        combined_graph, data_graph = combine_control_and_data_adjacents(adjacency_list)
        array_slices = []
        array_slices_bdir = []
        call_slices = []
        call_slices_bdir = []
        arith_slices = []
        arith_slices_bdir = []
        ptr_slices = []
        ptr_slices_bdir = []
        all_slices = []
        all_slices_vd = []

        all_keys = set()
        _keys = set()
        for slice_ln in call_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))

            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))


            call_slices.append(vdp_lines)
            call_slices_bdir.append(all_slice_lines)

            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)


        _keys = set()
        for slice_ln in array_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))

            array_slices.append(vdp_lines)
            array_slices_bdir.append(all_slice_lines)
            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)

        _keys = set()
        for slice_ln in arithmatic_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))
            arith_slices.append(vdp_lines)
            arith_slices_bdir.append(all_slice_lines)
            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)

        _keys = set()
        for slice_ln in ptr_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))
            ptr_slices.append(vdp_lines)
            ptr_slices_bdir.append(all_slice_lines)
            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)

        data_instance = {
            'file_path': cpg,
            'call_slices_vd': call_slices,
            'call_slices_sy': call_slices_bdir,
            'array_slices_vd': array_slices,
            'array_slices_sy': array_slices_bdir,
            'arith_slices_vd': arith_slices,
            'arith_slices_sy': arith_slices_bdir,
            'ptr_slices_vd': ptr_slices,
            'ptr_slices_sy': ptr_slices_bdir,
            'all_slices_sy': all_slices,
            'all_slices_vd': all_slices_vd
        }
        all_data.append(data_instance)
    return all_data

def get_data_d2a(root):
   
    sensi_api_path = "../resources/sensiAPI.txt"
    cpg_list = [root]


    with open(sensi_api_path, "r", encoding="utf-8") as f:
        sensi_api_set = set([api.strip() for api in f.read().split(",")])


    all_data = list()
    for cpg in cpg_list:
        nodes_path = join(cpg, "nodes.csv")
        edges_path = join(cpg, "edges.csv")
        with open(nodes_path, "r") as f:
            nodes = [node for node in csv.DictReader(f, delimiter='\t')]
        call_lines = set()
        array_lines = set()
        ptr_lines = set()
        arithmatic_lines = set()
        if len(nodes) == 0:
            continue
        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            if ntype == 'CallExpression':
                function_name = nodes[node_idx + 1]['code']
                if function_name is None or function_name.strip() == '':
                    continue
                if function_name.strip() in sensi_api_set:
                    line_no = extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            elif ntype == 'ArrayIndexing':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            elif ntype == 'PtrMemberAccess':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            elif node['operator'].strip() in ['+', '-', '*', '/']:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)


        nodes = read_csv(nodes_path)
        edges = read_csv(edges_path)
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges,
                                               False)

        combined_graph, data_graph = combine_control_and_data_adjacents(adjacency_list)
        array_slices = []
        array_slices_bdir = []
        call_slices = []
        call_slices_bdir = []
        arith_slices = []
        arith_slices_bdir = []
        ptr_slices = []
        ptr_slices_bdir = []
        all_slices = []
        all_slices_vd = []

        all_keys = set()
        _keys = set()
        for slice_ln in call_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))

            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))


            call_slices.append(vdp_lines)
            call_slices_bdir.append(all_slice_lines)

            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)


        _keys = set()
        for slice_ln in array_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))

            array_slices.append(vdp_lines)
            array_slices_bdir.append(all_slice_lines)
            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)

        _keys = set()
        for slice_ln in arithmatic_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))
            arith_slices.append(vdp_lines)
            arith_slices_bdir.append(all_slice_lines)
            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)

        _keys = set()
        for slice_ln in ptr_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
            vdp_for_lines = create_forward_slice(data_graph, slice_ln)
            vdp_bk_lines.extend(vdp_for_lines)
            vdp_lines = sorted(list(set(vdp_bk_lines)))
            ptr_slices.append(vdp_lines)
            ptr_slices_bdir.append(all_slice_lines)
            all_slices.append(all_slice_lines)
            all_slices_vd.append(vdp_lines)

        data_instance = {
            'file_path': cpg,
            'call_slices_vd': call_slices,
            'call_slices_sy': call_slices_bdir,
            'array_slices_vd': array_slices,
            'array_slices_sy': array_slices_bdir,
            'arith_slices_vd': arith_slices,
            'arith_slices_sy': arith_slices_bdir,
            'ptr_slices_vd': ptr_slices,
            'ptr_slices_sy': ptr_slices_bdir,
            'all_slices_sy': all_slices,
            'all_slices_vd': all_slices_vd
        }
        all_data.append(data_instance)
    return all_data

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_slice_for_cdg(info_dict, output_dir, abs_file_path):
    
    sensi_api_path = CUR_DIR + "/resources/sensiAPI.txt"
    


    with open(sensi_api_path, "r", encoding="utf-8") as f:
        sensi_api_set = set([api.strip() for api in f.read().split(",")])

    
    path = info_dict['path']
    vul_line = info_dict['line']

    src = os.path.join(abs_file_path, path)
    csv_root = output_dir + src
    
    nodes_path = join(csv_root, "nodes.csv")
    edges_path = join(csv_root, "edges.csv")
    if not os.path.exists(nodes_path):
        return None
    with open(nodes_path, "r") as f:
        nodes = [node for node in csv.DictReader(f, delimiter='\t')]
    call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()
    if len(nodes) == 0:
        return 
    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name is None or function_name.strip() == '':
                continue
            if function_name.strip() in sensi_api_set:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    call_lines.add(line_no)
        elif ntype == 'ArrayIndexing':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                array_lines.add(line_no)
        elif ntype == 'PtrMemberAccess':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                ptr_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                arithmatic_lines.add(line_no)
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
    adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges,
                                           False)
    combined_graph, data_graph = combine_control_and_data_adjacents(adjacency_list)
    array_slices = []
    array_slices_bdir = []
    call_slices = []
    call_slices_bdir = []
    arith_slices = []
    arith_slices_bdir = []
    ptr_slices = []
    ptr_slices_bdir = []
    all_slices = []
    all_slices_vd = []
    all_keys = set()
    _keys = set()
    for slice_ln in call_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
        vdp_for_lines = create_forward_slice(data_graph, slice_ln)
        vdp_bk_lines.extend(vdp_for_lines)
        vdp_lines = sorted(list(set(vdp_bk_lines)))
        call_slices.append(vdp_lines)
        call_slices_bdir.append(all_slice_lines)
        all_slices.append(all_slice_lines)
        all_slices_vd.append(vdp_lines)
    _keys = set()
    for slice_ln in array_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
        vdp_for_lines = create_forward_slice(data_graph, slice_ln)
        vdp_bk_lines.extend(vdp_for_lines)
        vdp_lines = sorted(list(set(vdp_bk_lines)))
        array_slices.append(vdp_lines)
        array_slices_bdir.append(all_slice_lines)
        all_slices.append(all_slice_lines)
        all_slices_vd.append(vdp_lines)
    _keys = set()
    for slice_ln in arithmatic_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
        vdp_for_lines = create_forward_slice(data_graph, slice_ln)
        vdp_bk_lines.extend(vdp_for_lines)
        vdp_lines = sorted(list(set(vdp_bk_lines)))
        arith_slices.append(vdp_lines)
        arith_slices_bdir.append(all_slice_lines)
        all_slices.append(all_slice_lines)
        all_slices_vd.append(vdp_lines)
    _keys = set()
    for slice_ln in ptr_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        vdp_bk_lines = create_backward_slice(data_graph, slice_ln)
        vdp_for_lines = create_forward_slice(data_graph, slice_ln)
        vdp_bk_lines.extend(vdp_for_lines)
        vdp_lines = sorted(list(set(vdp_bk_lines)))
        ptr_slices.append(vdp_lines)
        ptr_slices_bdir.append(all_slice_lines)
        all_slices.append(all_slice_lines)
        all_slices_vd.append(vdp_lines)

    file_content = [] 
    with open(src, 'r', encoding='utf8', errors="ignore") as f:
        file_content = f.readlines()

    data_instance = {
        'file_path': path,
        'call_slices_vd': call_slices,
        'call_slices_sy': call_slices_bdir,
        'array_slices_vd': array_slices,
        'array_slices_sy': array_slices_bdir,
        'arith_slices_vd': arith_slices,
        'arith_slices_sy': arith_slices_bdir,
        'ptr_slices_vd': ptr_slices,
        'ptr_slices_sy': ptr_slices_bdir,
        'all_slices_sy': all_slices,
        'all_slices_vd': all_slices_vd,
        'file_content' : file_content
    }

    return data_instance