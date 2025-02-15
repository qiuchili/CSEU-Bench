import random
from utils.const import (
    ALL_TASKS_CSU, 
    ALL_MODELS_CSU,
    TASK_LABELS_CSU
)
import json
import re

def reorder_performance_table_csu(df):
    ordered_columns = []
    ordered_rows = []
    for task in ALL_TASKS_CSU:
        if task in df.columns:
            ordered_columns.append(task)
    for model in ALL_MODELS_CSU:
        if model in df.index:
            ordered_rows.append(model)
    df = df[ordered_columns].reindex(ordered_rows)
    return df
    
def get_random_shuffled_labels_str(task_name_or_labels):
    if isinstance(task_name_or_labels, str):
       labels = list(TASK_LABELS_CSU[task_name_or_labels].values())
    else:
        labels = list(task_name_or_labels.values())
    
    random.shuffle(labels)
    all_label_str = ', '.join([f'"{l}"' for l in labels])
    return all_label_str

def postprocess_llm_output(target_str, labels):
    if not isinstance(target_str, str):
        return target_str, ''
    output_str = target_str.split('### 你的回答:')[-1].strip()
    processed_output_str = re.split(r'\n+',output_str)[0]
    
    # Extract label from output_str

    if len(labels) == 0:
        return output_str
    
    for k, v in labels.items():
        # Consider multiple entities split by '/'
        for vv in v.split('/'):
            if vv in processed_output_str:
                return output_str, k
    return output_str, ''  

def eval_performance(y_true, y_pred, label_clustering_tree,metric_path=None):
    from utils.label_clustering import get_distance
    sum_acc = 0
    sum_binary_acc = 0 
    for y_t, y_p in zip(y_true, y_pred):
        distance = get_distance(y_t, y_p, label_clustering_tree)
        sum_acc += 1-distance
        sum_binary_acc += 1 if y_t == y_p else 0

    acc = sum_acc/len(y_true)
    sum_binary_acc = sum_binary_acc/len(y_true)
    metric_dict = {'Accuracy': acc, 'Binary Accuracy': sum_binary_acc}
    if metric_path is not None:
       json.dump(metric_dict,open(metric_path,'w'),indent=4)
    
    # print(acc)
    return metric_dict