# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import pdb
from multiprocessing import Pool
import linecache
import argparse
from scipy import sparse

"""
python process_data.py -p 4 -b 1000000 a.txt b.txt c.txt
"""
parser = argparse.ArgumentParser(description="python process_data.py -p 4 -b 1000000 a.txt b.txt c.txt")
parser.add_argument("--process","-p",type=int,default=2)
parser.add_argument("--block_size","-b",type=int,default=100000)
parser.add_argument("--num_features","-n",type=int,default=1355192)
parser.add_argument("--format","-f",type=str,default="fm")
parser.add_argument("--array",action="store_true",help="store data as np.array")
parser.add_argument("--filenames",nargs="+",type=str,default='C:\\Users\\asd\\Desktop\\数据集\\news20\\news20.binary.txt')

results = {}

def parse_line_fm(line):
    line = line.strip().split()
    if len(line) <= 1:
        return None,None,None,None
    label = np.float32(line[0])
    line_data = np.array([l.split(":") for l in line[1:]])
    feat_idx = line_data[:,0].astype(np.int32)
    vals = line_data[:,1].astype(np.float32)
    return None,feat_idx,vals,label

def parse_line_ffm(line):
    line = line.strip().split()
    if len(line) <= 1:
        return None,None,None,None
    label = np.float32(line[0])
    line_data = np.array([l.split(":") for l in line[1:]])
    field_idx = line_data[:,0].astype(np.int32)
    feat_idx = line_data[:,1].astype(np.int32)
    vals  = line_data[:,2].astype(np.float32)
    return field_idx,feat_idx,vals,label

def work(parse_func,data,num_features,part_name,use_array=False):
    """Subprocess works.
    Args:
        parse_func: function to parse lines, support "ffm" and "fm" formats.
        data: raw data wait to be processed.
        parts_name: the total raw data is split into several parts, ranked by their index.
    """
    print("task {} starts.".format(part_name))
    rows = []
    cols = []
    values = []
    labels = []
    row_offset = 0
    # parse lines
    for row,line in enumerate(data):
        if row % 10000 == 0:
            print("processing {} in {}".format(row, part_name))
        _,col,val,label =  parse_func(line)
        if label is None:
            row_offset += 1
            continue
        rows.extend([row - row_offset]*len(col))
        values.extend(val)
        cols.extend(col)
        labels.append(label)

    data = sparse.csc_matrix((values,(rows,cols)),shape=(len(data)-row_offset,num_features))
    if use_array:
        data = data.toarray()
        
    print("task {} ends.".format(part_name))
    return part_name, data, labels

def process_res_list(res_list):
    for res in res_list:
        part_name, sp_data, labels = res.get()
        print("Part name", part_name)
        results[part_name] = {}
        results[part_name]["data"] = sp_data
        results[part_name]["label"] = np.array(labels).flatten().astype(int)

def post_process(filenames,use_array=False):
    """Merge each files parts together.
    """
    start_time = time.time()
    print("Postprocessing..")
    for file in filenames:
        data_list = []
        index_list = []
        for k,v in results.items():
            base_name, index = k.split("::")
            if base_name == file:
                index_list.append(int(index))
                data_list.append(v)
        total_data = None
        sorted_index = np.argsort(index_list)
        for i in sorted_index:
            if total_data is None:
                total_data = {}
                total_data["data"] = data_list[i]["data"]
                total_data["label"] = data_list[i]["label"]
            else:
                if not use_array:
                    total_data["data"] = sparse.vstack([total_data["data"],data_list[i]["data"]])
                else:
                    total_data["data"] = np.r_[total_data["data"],data_list[i]["data"]]
                total_data["label"] = np.r_[total_data["label"],data_list[i]["label"]]

        filename = "{}.npy".format(file)
        duration = time.time() - start_time
        print("Save {}, cost {:.1f} sec.".format(filename,duration))
        np.save(filename,total_data)

    return


args = parser.parse_args()
filenames = args.filenames
num_processes = args.process
block_size = args.block_size
num_features = args.num_features
data_format = args.format
use_arrsay = args.array

# unit test
assert block_size > 0
assert num_features > 0
assert num_processes > 0
assert data_format in ["fm", "ffm"]

if data_format == "fm":
    parse_func = parse_line_fm
elif data_format == "ffm":
    parse_func = parse_line_ffm

parse_func = parse_line_fm
file = 'C:\\Users\\dell\\Desktop\\数据集\\Skin.txt'
raw_data = linecache.getlines(file)
rows = []
cols = []
values = []
labels = []
row_offset = 0

for row, line in enumerate(raw_data):
    _, col, val, label = parse_func(line)
    if label is None:
        row_offset += 1
        continue
    rows.extend([row - row_offset] * len(col))
    values.extend(val)
    cols.extend(col)
    labels.append(label)

data = sparse.csc_matrix((values,(rows,cols)),shape=(len(raw_data)-row_offset,num_features))
print(data)
results["data"] = data
results["label"] = np.array(labels)
name = 'skin'
filename = "{}.npy".format(name)
np.save(filename,results)
print(labels)
"""
_, sp_data, label = work(parse_func, raw_data, num_features, "{}::0".format(file), use_arrsay)
results["{}::0".format(file)] = {}
results["{}::0".format(file)]["data"] = sp_data
results["{}::0".format(file)]["label"] = np.array(label)
print("{} done.".format(file))
print(label)
"""








