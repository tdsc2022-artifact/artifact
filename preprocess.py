from argparse import ArgumentParser
from utils.common import get_config
from preprocessing import generate_bc_VFG, generate_CDG
from preprocessing import cdg_preprocess
from preprocessing.cdg_generator import generate_CDG, generate_CDG_osp
from preprocessing.cfg_generator import generate_CFG, generate_CFG_osp
from preprocessing.dot_generator import generate_bc_VFG, generate_bc_VFG_osp, generate_VFG
import os
# from utils.common import CWEID_AVA, CWEID_ADOPT
from preprocessing import token_preprocess
from preprocessing import c2s_preprocess
from utils.xml_parser import  getCodeIDtoPathDict_osp, create_osp_source_code ,create_d2a_source_code
from os.path import join
import xml.etree.ElementTree as ET
import json
if __name__ == "__main__":
   
    config = get_config("code2vec","CWE20")
    preprocess_d2a('code2vec', 'NULLPTR_DEREFERENCE')
    