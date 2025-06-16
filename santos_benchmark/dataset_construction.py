import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

folder_datalake = '../santos_benchmark_ori/datalake'
folder_query = '../santos_benchmark_ori/query'
csv_files = [f for f in os.listdir(folder_datalake) if f.endswith('.csv')]
csv_files_query = [f for f in os.listdir(folder_query) if f.endswith('.csv')]

entity_matching_labels = {"ltable_name":[], "l_no":[], "rtable_name":[], "r_no":[], "label":[]}
schema_matching_labels = {"table_name":[], "ori_column_name":[], "column_idx":[], "renamed_column_name":[]}
unionable_table_search_labels = {"table_name":[], "dataset_name":[], "unionable_id":[]}
joinable_table_search_labels = {"table_name_1":[], "column_name_1":[], "table_name_2":[], "column_name_2":[], "ratio":[]}


mapping = {}
cnt = 0

for file in csv_files:

    key = "_".join(file.split('_')[0:-1])
    if key in mapping.keys():
        mapping[key].append(f"{cnt}_{file}")
    else:
        mapping[key] = [f"{cnt}_{file}"]

    # generation only for one time
    # df = pd.read_csv(os.path.join(folder_datalake, file))
    # df.to_csv(f"./datalake/{cnt}_{file}", index=False)
    cnt += 1

for file in csv_files_query:

    key = "_".join(file.split('_')[0:-1])
    if key in mapping.keys():
        mapping[key].append(f"{cnt}_{file}")
    else:
        mapping[key] = [f"{cnt}_{file}"]

    # generation only for one time
    # df = pd.read_csv(os.path.join(folder_query, file))
    # df.to_csv(f"./datalake/{cnt}_{file}", index=False)
    cnt += 1

import sqlite3
from itertools import product

def compute_overlap_ratio(basepath, df1_name, df2_name):
    # 假设两个文件是 file1.csv 和 file2.csv
    df1 = pd.read_csv(os.path.join(basepath, df1_name))
    df2 = pd.read_csv(os.path.join(basepath, df2_name))

    joinable_pairs = []

    for col1 in df1.columns:
        for col2 in df2.columns:
            set1 = set(df1[col1].dropna().unique())
            set2 = set(df2[col2].dropna().unique())
            if not set1 or not set2:
                continue
            overlap = set1 & set2
            ratio = len(overlap) / (len(set1)+len(set2)+len(overlap))
            joinable_pairs.append([col1, col2, ratio])
    
    return joinable_pairs


folder_path = './datalake'
unionable_id = 0
for key in tqdm(mapping.keys()):
    # print("=================================")
    # print(f"the key is {key}")
    if key is None:
        continue
    if len(mapping[key]) < 2:
        continue
    joinable_candidates = mapping[key]
    # print(joinable_candidates)
    for i in tqdm(range(len(joinable_candidates))):
        for j in range(i+1, len(joinable_candidates)):
            # print(joinable_candidates[i], joinable_candidates[j])
            joinable_pairs = compute_overlap_ratio(folder_path, joinable_candidates[i], joinable_candidates[j])
            # print(joinable_pairs)
            for m in range(len(joinable_pairs)):
                joinable_table_search_labels["table_name_1"].append(joinable_candidates[i])
                joinable_table_search_labels["column_name_1"].append(joinable_pairs[m][0])
                joinable_table_search_labels["table_name_2"].append(joinable_candidates[j])
                joinable_table_search_labels["column_name_2"].append(joinable_pairs[m][1])
                joinable_table_search_labels["ratio"].append(joinable_pairs[m][2])

    for i in range(len(mapping[key])):
        unionable_table_search_labels["table_name"].append(mapping[key][i])
        unionable_table_search_labels["dataset_name"].append(key)
        unionable_table_search_labels["unionable_id"].append(unionable_id)
    unionable_id += 1


with open('output_concise_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for key in tqdm(data.keys()):
        if int(len(key.split('_'))/2)*2 != len(key.split('_')):
            continue
        
        dataset_name = '_'.join(key.split('_')[0:int(len(key.split('_'))/2)-1])
        table_name = '_'.join(key.split('_')[int(len(key.split('_'))/2)-1:])
        table_name_without_cnt = '_'.join(key.split('_')[int(len(key.split('_'))/2):])
        print(dataset_name, table_name)

        try:
            df = pd.DataFrame(data[key]["entity"], columns=data[key]["column_name"])
        except ValueError as e:
            print(f"Skip {key} due to mismatched columns: {e}")
            continue  # 跳过当前循环
        
        ori_df = pd.read_csv(f"./datalake/{table_name}", encoding='utf-8')
        ori_columns = ori_df.columns.tolist()
        for idx, column in enumerate(df.columns):
            schema_matching_labels["table_name"].append(f"{cnt}_{table_name_without_cnt}")
            schema_matching_labels["ori_column_name"].append(ori_columns[idx])
            schema_matching_labels["column_idx"].append(idx)
            schema_matching_labels["renamed_column_name"].append(column)

        for i, idx in enumerate(data[key]["original_selected_index"]):
            entity_matching_labels["ltable_name"].append(table_name)
            entity_matching_labels["l_no"].append(idx)
            entity_matching_labels["rtable_name"].append(f"{cnt}_{table_name_without_cnt}")
            entity_matching_labels["r_no"].append(i)
            entity_matching_labels["label"].append(1)
        
        unionable_table_search_labels["table_name"].append(f"{cnt}_{table_name_without_cnt}")
        unionable_table_search_labels["dataset_name"].append(dataset_name)
        unionable_table_search_labels["unionable_id"].append(unionable_id)
        unionable_table_search_labels["table_name"].append(table_name)
        unionable_table_search_labels["dataset_name"].append(dataset_name)
        unionable_table_search_labels["unionable_id"].append(unionable_id)
        unionable_id += 1

        df.to_csv(f'./datalake/{cnt}_{table_name_without_cnt}', index=False)
        cnt += 1

labels_df = pd.DataFrame(entity_matching_labels)
labels_file_path = os.path.join("./label/entity_matching/", "entity_matching_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')

labels_df = pd.DataFrame(schema_matching_labels)
labels_file_path = os.path.join("./label/schema_matching/", "schema_matching_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')

labels_df = pd.DataFrame(unionable_table_search_labels)
labels_file_path = os.path.join("./label/unionable_table_search/", "unionable_table_search_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')

labels_df = pd.DataFrame(joinable_table_search_labels)
labels_file_path = os.path.join("./label/joinable_table_search/", "joinable_table_search_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')

