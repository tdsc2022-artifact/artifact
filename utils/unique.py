'''unique json

@author : jumormt
@version : 1.0
'''
import hashlib


def getMD5(s):
    '''
    得到字符串s的md5加密后的值

    :param s:
    :return:
    '''
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()


def unique_token(token_list):
    md5Dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for token in token_list:
        target = token["target"]
        content = token["content"]
        token_md5 = getMD5(str(content))
        if token_md5 not in md5Dict:
            md5Dict[token_md5] = dict()
            md5Dict[token_md5]["target"] = target
            md5Dict[token_md5]["content"] = content
        else:
            md5Target = md5Dict[token_md5]["target"]
            if (md5Target != -1 and md5Target != target):
                conflict_ct += 1
                md5Dict[token_md5]["target"] = -1
            else:
                mul_ct += 1
    print(f"total conflict: {conflict_ct}")
    print(f"total multiple: {mul_ct}")

    return md5Dict


def unique_cfg_list(cfg_list):
    md5Dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for cfg in cfg_list:
        target = cfg["target"]
        nodes_line = cfg["nodes"]
        nodes_line_md5 = list()
        for nls in nodes_line:
            nodes_line_md5.append(getMD5(str(nls)))
        edges = cfg["edges"]
        edges_md5 = list()
        for edge in edges:
            edges_md5.append(
                [nodes_line_md5[edge[0]], nodes_line_md5[edge[1]]])
        edges_md5 = sorted(edges_md5)
        cfgMD5 = getMD5(str(edges_md5))  # md5 all edges - cfg
        if cfgMD5 not in md5Dict.keys():
            md5Dict[cfgMD5] = dict()
            md5Dict[cfgMD5]["target"] = target
            md5Dict[cfgMD5]["cfg"] = cfg
        else:  # conflict - mark as -1
            md5Target = md5Dict[cfgMD5]["target"]
            if (md5Target != -1 and md5Target != target):
                conflict_ct += 1
                md5Dict[cfgMD5]["target"] = -1
            else:
                mul_ct += 1
    print(f"total conflict: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5Dict