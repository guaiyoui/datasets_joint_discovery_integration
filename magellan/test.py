# import json
# import numpy as np
# import pandas as pd
# np.random.seed(42)  # set seed for reproducibility

# with open('output_concise.json', 'r', encoding='utf-8') as f:
#     column_mapping_data = json.load(f)

# print(column_mapping_data["abt_buy_tableB.csv: id"]['alternative_column_names'])

# print(column_mapping_data.keys())



# print(np.random.choice(column_mapping_data["abt_buy_tableB.csv: id"]['alternative_column_names'], 1))

# df = pd.read_csv('../magellan_ori/abt_buy/tableA.csv')
# print(df.head())

# column_renamed = {}

# for column in df.columns.tolist():
#     print(column)
#     column_renamed[column] = np.random.choice(column_mapping_data[f"abt_buy_tableA.csv: {column}"]['alternative_column_names'], 1)[0]
#     print(column_renamed[column])

# print(df.columns.tolist().index("description"))


# df = df.rename(columns=column_renamed)

# print(df.head())

import pandas as pd
import sqlite3
from itertools import product
import os

def compute_overlap_ratio(basepath, df1_name, df2_name):
    # 假设两个文件是 file1.csv 和 file2.csv
    df1 = pd.read_csv(os.path.join(basepath, df1_name))
    df2 = pd.read_csv(os.path.join(basepath, df2_name))

    # 建立 SQLite 内存数据库（也可以用本地文件）
    conn = sqlite3.connect(':memory:')
    df1.to_sql('table1', conn, index=False, if_exists='replace')
    df2.to_sql('table2', conn, index=False, if_exists='replace')

   

    # 获取列名
    cols1 = df1.columns
    cols2 = df2.columns

    # 阈值：最小 overlap ratio（可调）
    threshold = 0.1  # 表示至少 10% 的值在对方中出现

    joinable_pairs = []

    for col1, col2 in product(cols1, cols2):
        query = f"""
        SELECT COUNT(*) as overlap_count FROM (
            SELECT DISTINCT t1."{col1}" as val FROM table1 t1
            INNER JOIN table2 t2 ON t1."{col1}" = t2."{col2}"
            WHERE t1."{col1}" IS NOT NULL AND t2."{col2}" IS NOT NULL
        )
        """
        cursor = conn.execute(query)
        overlap_count = cursor.fetchone()[0]
        
        unique1 = df1[col1].dropna().nunique()
        unique2 = df2[col2].dropna().nunique()
        
        if unique1 == 0 or unique2 == 0:
            continue

        ratio = overlap_count / min(unique1, unique2)
        
        
        joinable_pairs.append([col1, col2, ratio])
    return joinable_pairs


folder_path = './datalake'

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

unionable_table_search_labels = {"table_name":[], "dataset_name":[], "unionable_id":[]}
joinable_table_search_labels = {"table_name_1":[], "column_name_1":[], "table_name_2":[], "column_name_2":[], "ratio":[]}

# for fname in csv_files:
#     print(fname)
dataset_name = ["abt_buy", "amazon_google", "beer", "company", "dblp_acm", "dblp_scholar", "fodors_zagat", "itunes_amazon", "walmart_amazon"]

for i in range(len(csv_files)):
    for k in range(len(dataset_name)):
        if dataset_name[k] in csv_files[i]:
            unionable_table_search_labels["table_name"].append(csv_files[i])
            unionable_table_search_labels["dataset_name"].append(dataset_name[k])
            unionable_table_search_labels["unionable_id"].append(k)

    for j in range(i+1, len(csv_files)):
        print(csv_files[i], csv_files[j])
        joinable_pairs = compute_overlap_ratio(folder_path, csv_files[i], csv_files[j])
        print(joinable_pairs)
        for m in range(len(joinable_pairs)):
            joinable_table_search_labels["table_name_1"].append(csv_files[i])
            joinable_table_search_labels["column_name_1"].append(joinable_pairs[m][0])
            joinable_table_search_labels["table_name_2"].append(csv_files[j])
            joinable_table_search_labels["column_name_2"].append(joinable_pairs[m][1])
            joinable_table_search_labels["ratio"].append(joinable_pairs[m][2])
        break
    break

print(unionable_table_search_labels)
print(joinable_table_search_labels)
    

# 输出 joinable 列对
# for col1, col2, ratio in sorted(joinable_pairs, key=lambda x: -x[2]):
    # print(f"{col1} ↔ {col2}: overlap ratio = {ratio:.2f}")