# 给定一个training_ratio, validate_ration, testing_ratio. 请帮我实现train_validate_test.py这个文件, 该功能是将数据集划分成train, validate, test三个部分。每一个文件夹下分别存为train.csv, validate.csv, test.csv. 总共有四个数据集，其处理方法如下：(1) entity_matching. 其column name为ltable_name,l_id,rtable_name,r_id,label，随机分成对应的比例; (2) joinable_table_search. 其column_names为table_name_1,column_name_1,table_name_2,column_name_2,ratio. 请把ratio大于0.5的当作label=1, 然后把ratio小于0.05的当作label=0，其他就不要了，随机分成对应的比例，确保通过随机drop一些label=0的records，来确保label=0的records数量不超过label=1的record的3倍; (3) schema_matching. 其column_names为table_name,ori_column_name,column_idx,renamed_column_name. 请把每一个record的table_name用.split("_"), 然后再join第二个和第三个构建新的table name。如果两个record，这样子操作之后的table_name一样，同时ori_column_name一样，那么就是label=1，否则就是label=0; 最后的结果应该是table_name_1,renamed_column_name_1,table_name_2,renamed_column_name_2,label; 确保通过随机drop一些label=0，来确保label=0的records数量不超过label=1的record的3倍;(4) unionable_table_search. 其column_names是table_name,dataset_name,unionable_id. 如果任意两个records，他们的unionable_id是一样的话，请把label=1，否则就是label=0。 最后的结果应该是table_name_1,table_name_2,label; 确保通过随机drop一些label=0，来确保label=0的records数量不超过label=1的record的3倍;


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

def split_dataset(df, training_ratio, validate_ratio, testing_ratio):
    """
    将数据集按照给定比例划分为训练集、验证集和测试集
    """
    # 确保比例之和为1
    total_ratio = training_ratio + validate_ratio + testing_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和应为1.0，当前为 {total_ratio}")
    
    # 首先分离出测试集
    train_val_df, test_df = train_test_split(
        df, 
        test_size=testing_ratio, 
        random_state=42,
        stratify=df['label'] if 'label' in df.columns else None
    )
    
    # 然后从剩余数据中分离出训练集和验证集
    relative_val_size = validate_ratio / (training_ratio + validate_ratio)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        random_state=42,
        stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
    )
    
    return train_df, val_df, test_df

def balance_labels(df, max_ratio=3):
    """
    确保label=0的记录数量不超过label=1记录数量的max_ratio倍
    """
    if 'label' not in df.columns:
        return df
    
    label_1_count = len(df[df['label'] == 1])
    label_0_count = len(df[df['label'] == 0])
    
    if label_0_count > label_1_count * max_ratio:
        # 随机删除一些label=0的记录
        label_0_indices = df[df['label'] == 0].index
        keep_count = label_1_count * max_ratio
        keep_indices = np.random.choice(label_0_indices, size=keep_count, replace=False)
        
        # 保留所有label=1的记录和部分label=0的记录
        final_indices = df[df['label'] == 1].index.tolist() + keep_indices.tolist()
        df = df.loc[final_indices].reset_index(drop=True)
    
    return df

def process_entity_matching(df):
    """
    处理实体匹配数据集
    输入列: ltable_name, l_id, rtable_name, r_id, label
    """
    # 检查必要的列是否存在
    required_cols = ['ltable_name', 'l_id', 'rtable_name', 'r_id', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    return df[required_cols].copy()

def process_joinable_table_search(df):
    """
    处理可连接表搜索数据集
    输入列: table_name_1, column_name_1, table_name_2, column_name_2, ratio
    """
    required_cols = ['table_name_1', 'column_name_1', 'table_name_2', 'column_name_2', 'ratio']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 根据ratio生成label
    df['label'] = 0
    df.loc[df['ratio'] > 0.1, 'label'] = 1
    
    # 只保留ratio > 0.1 或 ratio < 0.05的记录
    df_filtered = df[(df['ratio'] > 0.1) | (df['ratio'] < 0.05)].copy()
    
    # 平衡标签
    df_balanced = balance_labels(df_filtered)
    
    # 返回最终列
    return df_balanced[['table_name_1', 'column_name_1', 'table_name_2', 'column_name_2', 'label']]

def process_schema_matching(df):
    """
    处理模式匹配数据集
    输入列: table_name, ori_column_name, column_idx, renamed_column_name
    """
    required_cols = ['table_name', 'ori_column_name', 'column_idx', 'renamed_column_name']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 构建新的table_name
    df['new_table_name'] = df['table_name'].apply(lambda x: '_'.join(x.split('_')[1:3]) if len(x.split('_')) >= 3 else x)
    
    # 生成所有可能的记录对
    records = []
    df_list = df.to_dict('records')
    
    for i, record1 in enumerate(df_list):
        for j, record2 in enumerate(df_list):
            if i >= j:  # 避免重复和自己与自己比较
                continue
                
            # 判断是否为正样本
            if (record1['new_table_name'] == record2['new_table_name'] and 
                record1['ori_column_name'] == record2['ori_column_name']):
                label = 1
            else:
                label = 0
            
            records.append({
                'table_name_1': record1['table_name'],
                'renamed_column_name_1': record1['renamed_column_name'],
                'table_name_2': record2['table_name'],
                'renamed_column_name_2': record2['renamed_column_name'],
                'label': label
            })
    
    result_df = pd.DataFrame(records)
    
    # 平衡标签
    result_df = balance_labels(result_df)
    
    return result_df

def process_unionable_table_search(df):
    """
    处理可联合表搜索数据集
    输入列: table_name, dataset_name, unionable_id
    """
    required_cols = ['table_name', 'dataset_name', 'unionable_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 生成所有可能的记录对
    records = []
    df_list = df.to_dict('records')
    
    for i, record1 in enumerate(df_list):
        for j, record2 in enumerate(df_list):
            if i >= j:  # 避免重复和自己与自己比较
                continue
            
            # 判断是否为正样本
            if record1['unionable_id'] == record2['unionable_id']:
                label = 1
            else:
                label = 0
            
            records.append({
                'table_name_1': record1['table_name'],
                'table_name_2': record2['table_name'],
                'label': label
            })
    
    result_df = pd.DataFrame(records)
    
    # 平衡标签
    result_df = balance_labels(result_df)
    
    return result_df

def process_and_split_dataset(dataset_path, dataset_type, output_dir, training_ratio, validate_ratio, testing_ratio):
    """
    处理并划分单个数据集
    """
    # 读取数据
    df = pd.read_csv(dataset_path)
    print(f"原始数据集 {dataset_type} 大小: {len(df)}")
    
    # 根据数据集类型进行处理
    if dataset_type == 'entity_matching':
        processed_df = process_entity_matching(df)
    elif dataset_type == 'joinable_table_search':
        processed_df = process_joinable_table_search(df)
    elif dataset_type == 'schema_matching':
        processed_df = process_schema_matching(df)
    elif dataset_type == 'unionable_table_search':
        processed_df = process_unionable_table_search(df)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    print(f"处理后数据集 {dataset_type} 大小: {len(processed_df)}")
    
    # 如果有label列，打印标签分布
    if 'label' in processed_df.columns:
        label_counts = processed_df['label'].value_counts()
        print(f"标签分布: {dict(label_counts)}")
    
    # 划分数据集
    train_df, val_df, test_df = split_dataset(processed_df, training_ratio, validate_ratio, testing_ratio)
    
    # 创建输出目录
    # dataset_output_dir = os.path.join(output_dir, dataset_type)
    dataset_output_dir = f"./{dataset_type}"
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 保存文件
    train_df.to_csv(os.path.join(dataset_output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_output_dir, 'validate.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_output_dir, 'test.csv'), index=False)
    
    print(f"数据集 {dataset_type} 划分完成:")
    print(f"  训练集: {len(train_df)} 条记录")
    print(f"  验证集: {len(val_df)} 条记录")
    print(f"  测试集: {len(test_df)} 条记录")
    print(f"  文件保存至: {dataset_output_dir}")
    print("-" * 50)

def main():
    """
    主函数
    """
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 配置参数
    training_ratio = 0.5
    validate_ratio = 0.1
    testing_ratio = 0.4
    
    # 数据集路径配置（请根据实际情况修改）
    datasets = {
        'entity_matching': './entity_matching/entity_matching_labels.csv',
        'joinable_table_search': './joinable_table_search/joinable_table_search_labels.csv',
        'schema_matching': './schema_matching/schema_matching_labels.csv',
        'unionable_table_search': './unionable_table_search/unionable_table_search_labels.csv'
    }
    
    output_dir = 'output'
    
    # 处理每个数据集
    for dataset_type, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            try:
                process_and_split_dataset(
                    dataset_path=dataset_path,
                    dataset_type=dataset_type,
                    output_dir=dataset_type,
                    training_ratio=training_ratio,
                    validate_ratio=validate_ratio,
                    testing_ratio=testing_ratio
                )
            except Exception as e:
                print(f"处理数据集 {dataset_type} 时出错: {str(e)}")
        else:
            print(f"数据集文件不存在: {dataset_path}")
    
    print("所有数据集处理完成！")

# 如果需要自定义参数运行，可以使用以下函数
def run_with_custom_params(datasets_config, output_dir, training_ratio, validate_ratio, testing_ratio):
    """
    使用自定义参数运行数据集划分
    
    Args:
        datasets_config: dict, 数据集配置 {'dataset_type': 'file_path'}
        output_dir: str, 输出目录
        training_ratio: float, 训练集比例
        validate_ratio: float, 验证集比例
        testing_ratio: float, 测试集比例
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    for dataset_type, dataset_path in datasets_config.items():
        if os.path.exists(dataset_path):
            try:
                process_and_split_dataset(
                    dataset_path=dataset_path,
                    dataset_type=dataset_type,
                    output_dir=output_dir,
                    training_ratio=training_ratio,
                    validate_ratio=validate_ratio,
                    testing_ratio=testing_ratio
                )
            except Exception as e:
                print(f"处理数据集 {dataset_type} 时出错: {str(e)}")
        else:
            print(f"数据集文件不存在: {dataset_path}")

if __name__ == "__main__":
    main()