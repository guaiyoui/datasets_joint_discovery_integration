# folder_name = ["abt_buy", "amazon_google", "beer", "company", "dblp_acm", "dblp_scholar", "fodors_zagat", "itunes_amazon", "walmart_amazon"]

import pandas as pd
import numpy as np
import os
import bisect
import json
from tqdm import tqdm
np.random.seed(42)  # set seed for reproducibility

base_path = "../magellan_ori/"
output_folder = "../magellan/datalake/"

dataset_names = ["abt_buy", "amazon_google", "beer", "company", "dblp_acm", "dblp_scholar", "fodors_zagat", "itunes_amazon", "walmart_amazon"]

# chunk into tables


chunk_size_row = 100

def chunk_dataframe(df, chunk_size_row, chunk_size_col):
    if chunk_size_row < int(len(df)/5):
        chunk_size_row = int(len(df)/5)
    num_chunks = int(np.ceil(len(df) / chunk_size_row))
    chunks = []
    start_rows = []

    finish = False
    for i in range(num_chunks):
        start_row = i * chunk_size_row
        end_row = min((i + 1) * chunk_size_row, len(df))
        if (len(df) - end_row) < 0.5*chunk_size_row:
            end_row = len(df)
            finish = True
        df_chunk = df.iloc[start_row:end_row]
        start_rows.append(start_row)
        all_columns = df_chunk.columns.tolist()
        id_col = "id" if "id" in all_columns else None

        candidate_cols = [col for col in all_columns if col != id_col]
        selected_cols = list(np.random.choice(candidate_cols, min(chunk_size_col, len(candidate_cols)), replace=False))

         # 如果存在"id"，则把它加到最前面
        if id_col is not None:
            selected_cols = [id_col] + selected_cols

        df_selected = df_chunk[selected_cols]
        chunks.append(df_selected)

        if finish:
            break
    start_rows.append(len(df)+1)
    return chunks, start_rows

all_chunks = {}
table_cnt = 0

with open('output_concise.json', 'r', encoding='utf-8') as f:
    column_mapping_data = json.load(f)

entity_matching_labels = {"ltable_name":[], "l_id":[], "rtable_name":[], "r_id":[], "label":[]}

schema_matching_labels = {"table_name":[], "ori_column_name":[], "column_idx":[], "renamed_column_name":[]}

for dataset_name in dataset_names:
    print(f"Processing {dataset_name}.....")
    folder_path = os.path.join(base_path, dataset_name)
            
    for j in range(2):
        file_path = os.path.join(folder_path, "tableA.csv")
        df = pd.read_csv(file_path)
        chunk_size_col = df.shape[1]-1-j
        chunks, start_rows_A = chunk_dataframe(df, chunk_size_row, chunk_size_col)
        
        for idx, chunk in enumerate(chunks):
            base_filename = "tableA"
            chunk_filename = f"{table_cnt}_{dataset_name}_{base_filename}_colsize{chunk_size_col+1}_chunk{idx}.csv"
            chunk_path = os.path.join(output_folder, chunk_filename)

            column_renamed = {}
            # creat schema_matching_labels
            for column in df.columns.tolist():
                column_renamed[column] = np.random.choice(column_mapping_data[f"{dataset_name}_tableA.csv: {column}"]['alternative_column_names'], 1)[0]

                schema_matching_labels["table_name"].append(f"{table_cnt}_{dataset_name}_{base_filename}_colsize{chunk_size_col+1}_chunk{idx}.csv")
                schema_matching_labels["ori_column_name"].append(column)
                schema_matching_labels["column_idx"].append(df.columns.tolist().index(column))
                schema_matching_labels["renamed_column_name"].append(column_renamed[column])
            chunk = chunk.rename(columns=column_renamed)

            chunk.to_csv(chunk_path, index=False)
            table_cnt += 1
        

        file_path = os.path.join(folder_path, "tableB.csv")
        df = pd.read_csv(file_path)
        chunk_size_col = df.shape[1]-1-j
        chunks, start_rows_B = chunk_dataframe(df, chunk_size_row, chunk_size_col)
        
        for idx, chunk in enumerate(chunks):
            base_filename = "tableB"
            chunk_filename = f"{table_cnt}_{dataset_name}_{base_filename}_colsize{chunk_size_col+1}_chunk{idx}.csv"
            chunk_path = os.path.join(output_folder, chunk_filename)

            column_renamed = {}
            # creat schema_matching_labels
            for column in df.columns.tolist():
                column_renamed[column] = np.random.choice(column_mapping_data[f"{dataset_name}_tableA.csv: {column}"]['alternative_column_names'], 1)[0]

                schema_matching_labels["table_name"].append(f"{table_cnt}_{dataset_name}_{base_filename}_colsize{chunk_size_col+1}_chunk{idx}.csv")
                schema_matching_labels["ori_column_name"].append(column)
                schema_matching_labels["column_idx"].append(df.columns.tolist().index(column))
                schema_matching_labels["renamed_column_name"].append(column_renamed[column])
            chunk = chunk.rename(columns=column_renamed)

            chunk.to_csv(chunk_path, index=False)
            table_cnt += 1
        

        if j == 1:
            continue
            
        # create entity_matching_labels
        df_train = pd.read_csv(os.path.join(folder_path, "train.csv"))
        df_test = pd.read_csv(os.path.join(folder_path, "test.csv"))
        df_validate = pd.read_csv(os.path.join(folder_path, "valid.csv"))

        for df_target in [df_train, df_test, df_validate]:
            for idx in range(len(df_target)):

                ltable_id = df_target.iloc[idx]["ltable_id"]
                seq = bisect.bisect_right(start_rows_A, int(ltable_id))
                # print(f"table_cnt = {table_cnt}, start_rows_A={start_rows_A}, seq={seq}, ltable_cnt={ltable_id}")
                chunk_idx = seq-1
                ltable_cnt = table_cnt-len(start_rows_B)-len(start_rows_A)+2+seq-1
                entity_matching_labels["ltable_name"].append(f"{ltable_cnt}_{dataset_name}_tableA_colsize{chunk_size_col+1}_chunk{chunk_idx}.csv")
                entity_matching_labels["l_id"].append(df_target.iloc[idx]["ltable_id"])

                rtable_id = df_target.iloc[idx]["rtable_id"]
                seq = bisect.bisect_right(start_rows_B, int(rtable_id))
                chunk_idx = seq-1
                rtable_cnt = table_cnt-len(start_rows_B)+1+seq-1
                entity_matching_labels["rtable_name"].append(f"{rtable_cnt}_{dataset_name}_tableB_colsize{chunk_size_col+1}_chunk{chunk_idx}.csv")
                entity_matching_labels["r_id"].append(df_target.iloc[idx]["rtable_id"])

                entity_matching_labels["label"].append(df_target.iloc[idx]["label"])


# Save entity_matching_labels to a CSV file
labels_df = pd.DataFrame(entity_matching_labels)
labels_file_path = os.path.join("./label/entity_matching/", "entity_matching_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')


# save schema_matching_labels to a CSV file

labels_df = pd.DataFrame(schema_matching_labels)
labels_file_path = os.path.join("./label/schema_matching/", "schema_matching_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')


import sqlite3
from itertools import product

def compute_overlap_ratio(basepath, df1_name, df2_name):
    # 假设两个文件是 file1.csv 和 file2.csv
    df1 = pd.read_csv(os.path.join(basepath, df1_name))
    df2 = pd.read_csv(os.path.join(basepath, df2_name))

    # 建立 SQLite 内存数据库（也可以用本地文件）
    # conn = sqlite3.connect(':memory:')
    # df1.to_sql('table1', conn, index=False, if_exists='replace')
    # df2.to_sql('table2', conn, index=False, if_exists='replace')
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

    

    # for col1, col2 in product(cols1, cols2):
    #     query = f"""
    #     SELECT COUNT(*) as overlap_count FROM (
    #         SELECT DISTINCT t1."{col1}" as val FROM table1 t1
    #         INNER JOIN table2 t2 ON t1."{col1}" = t2."{col2}"
    #         WHERE t1."{col1}" IS NOT NULL AND t2."{col2}" IS NOT NULL
    #     )
    #     """
    #     cursor = conn.execute(query)
    #     overlap_count = cursor.fetchone()[0]
        
    #     unique1 = df1[col1].dropna().nunique()
    #     unique2 = df2[col2].dropna().nunique()
        
    #     if unique1 == 0 or unique2 == 0:
    #         continue

    #     ratio = overlap_count / min(unique1, unique2)
        
        # joinable_pairs.append([col1, col2, ratio])

    return joinable_pairs


folder_path = './datalake'

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

unionable_table_search_labels = {"table_name":[], "dataset_name":[], "unionable_id":[]}
joinable_table_search_labels = {"table_name_1":[], "column_name_1":[], "table_name_2":[], "column_name_2":[], "ratio":[]}

# for fname in csv_files:
#     print(fname)


for i in tqdm(range(len(csv_files))):
    for k in range(len(dataset_names)):
        if dataset_names[k] in csv_files[i]:
            unionable_table_search_labels["table_name"].append(csv_files[i])
            unionable_table_search_labels["dataset_name"].append(dataset_names[k])
            unionable_table_search_labels["unionable_id"].append(k)

    for j in range(i+1, len(csv_files)):
        # print(csv_files[i], csv_files[j])
        joinable_pairs = compute_overlap_ratio(folder_path, csv_files[i], csv_files[j])
        # print(joinable_pairs)
        for m in range(len(joinable_pairs)):
            joinable_table_search_labels["table_name_1"].append(csv_files[i])
            joinable_table_search_labels["column_name_1"].append(joinable_pairs[m][0])
            joinable_table_search_labels["table_name_2"].append(csv_files[j])
            joinable_table_search_labels["column_name_2"].append(joinable_pairs[m][1])
            joinable_table_search_labels["ratio"].append(joinable_pairs[m][2])

labels_df = pd.DataFrame(unionable_table_search_labels)
labels_file_path = os.path.join("./label/unionable_table_search/", "unionable_table_search_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')

labels_df = pd.DataFrame(joinable_table_search_labels)
labels_file_path = os.path.join("./label/joinable_table_search/", "joinable_table_search_labels.csv")
labels_df.to_csv(labels_file_path, index=False, encoding='utf-8')







