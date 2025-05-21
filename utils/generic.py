import random
from sklearn import metrics
from utils.const import (
    TASK_LABELS_CDI, 
    ALL_TASKS_CDI, 
    ALL_MODELS_CDI, 
    ALL_DATASETS_CSEU, 
    ALL_MODELS_CSEU,
    ATTITUDE_LABELS_CSEU,
    ALL_TASKS_CSEU
)
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re
import os
import pandas as pd
def reorder_performance_table_cdi(df):
    ordered_columns = []
    ordered_rows = []
    for task in ALL_TASKS_CDI:
        if task in df.columns:
            ordered_columns.append(task)
    for model in ALL_MODELS_CDI:
        if model in df.index:
            ordered_rows.append(model)
    df = df[ordered_columns].reindex(ordered_rows)
    return df

def reorder_performance_table_csu(df):
    ordered_columns = []
    ordered_rows = []
    for task in ALL_DATASETS_CSEU:
        if task in df.columns:
            ordered_columns.append(task)
    for model in ALL_MODELS_CSEU:
        if model in df.index:
            ordered_rows.append(model)
    df = df[ordered_columns].reindex(ordered_rows)
    return df

def reorder_performance_table_cseu(df):
    ordered_columns = []
    ordered_rows = []
    for task in ALL_TASKS_CSEU + ['all']:
        if task in df.columns:
            ordered_columns.append(task)
    for model in ALL_MODELS_CSEU:
        if model in df.index:
            ordered_rows.append(model)
    df = df[ordered_columns].reindex(ordered_rows)
    return df
    

def get_random_sampled_labels_str(task_name, true_label, n_labels=5):
    true_label_chinese = TASK_LABELS_CDI[task_name][true_label]
    labels = list(TASK_LABELS_CDI[task_name].values())
    labels.remove(true_label_chinese)
    selected_labels = random.choices(labels, k=n_labels-1)
    selected_labels.append(true_label_chinese)
    random.shuffle(selected_labels)
    all_label_str = ', '.join([f'"{l}"' for l in selected_labels])
    return all_label_str

def get_random_shuffled_labels_str(task_name_or_labels):
    if isinstance(task_name_or_labels, str):
       labels = list(TASK_LABELS_CDI[task_name_or_labels].values())
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

def majority_vote(input_list):
    count_dict = {k:input_list.count(k) for k in set(input_list)}
    max_count = max(count_dict.values())
    max_elements = [e for e in count_dict if count_dict[e] == max_count]
    return max_elements

def majority_vote_accuracy(human_preds, ground_truth):
    max_elements = majority_vote(human_preds)
    if ground_truth in max_elements:
        _acc = 1/len(max_elements)
    else:
        _acc = 0
    return _acc

def evaluate_multi_pred_performance(human_preds, ground_truth, labels):
    
    run_accs = np.array([sum(human_preds[:,i] == ground_truth)/len(ground_truth) for i in range(human_preds.shape[-1])])
    std_across_runs = run_accs.std()

    majority_vote_acc = sum([majority_vote_accuracy(list(_pred), _ground_truth) for _pred, _ground_truth in zip(human_preds, ground_truth)])/len(human_preds)
    acc = sum( (_pred == _ground_truth).mean() for _pred, _ground_truth in zip(human_preds, ground_truth))/len(human_preds)
    std = sum( (_pred == _ground_truth).std() for _pred, _ground_truth in zip(human_preds, ground_truth))/len(human_preds)
    one_hot_encoder = OneHotEncoder().fit([[l] for l in labels])
    ground_truth_rep = one_hot_encoder.transform([[g] for g in ground_truth]).toarray()
    preds_rep = np.stack([one_hot_encoder.transform([[s] for s in _pred]).toarray().mean(axis=0) for _pred in human_preds])
    ppl = -np.log(preds_rep[ground_truth_rep!=0] + 1e-10).mean()
    return {'majority_vote_acc': majority_vote_acc, 'acc':acc, 'std':std, 'ppl':ppl,'std_across_runs': std_across_runs}

def generate_evaluation_from_csv_file(all_models_folder= 'output_csu', model_name= 'gemini-pro-audio'):
    all_models_folder = 'output_csu'
    model_name = 'gemini-pro-audio'

    for dataset in ['1-七种情感库1男1女','2-相同文本的几种情绪-字面有无情绪','2-相同文本的几种情绪-第一部分',
                        '2-相同文本的几种情绪-第二部分', '4-情绪态度库']:
        for seed in [42, 666, 1234]:
            output_json_file = os.path.join(all_models_folder,model_name, f'eval_target_attitude_t_{dataset}_seed_{seed}.json')
            csv_file = os.path.join(all_models_folder,model_name, f'output_target_attitude_t_{dataset}_seed_{seed}.csv')
            df = pd.read_csv(csv_file,index_col=0)
            df['pred'] = df['llm_output'].apply(lambda x: postprocess_llm_output(x, ATTITUDE_LABELS_CSEU)[-1])
            df['target_attitude'] = df['target_attitude'].replace({'bs':'ng','cq':'cf','nq':'ng',
                                                                'yq':'yw','gq':'gx','hq':'hp',
                                                                'sq':'sq','xf':'gx'})

            df.to_csv(f'output_csu/qwen-audio-turbo/output_target_attitude_t_{dataset}_seed_{seed}.csv')

            all_labels = ATTITUDE_LABELS_CSEU
            print("Evaluation....")
            df = df[df['llm_output'] != '']
            label_to_id_dict = {k: i for i, (k, v) in enumerate(all_labels.items())}
            label_to_id_dict[''] = -1
            eval_performance(df['target_attitude'].map(label_to_id_dict), 
                                df['pred'].map(label_to_id_dict), 
                                output_json_file,silent=True)


def eval_performance_cseu(y_true, y_pred, label_clustering_tree,metric_path=None,max_depth=5):
    from utils.label_clustering import get_distance
    sum_acc = 0
    sum_binary_acc = 0 
    for y_t, y_p in zip(y_true, y_pred):
        distance = get_distance(y_t, y_p, label_clustering_tree,max_depth=max_depth)
        sum_acc += 1-distance
        sum_binary_acc += 1 if y_t == y_p else 0

    acc = sum_acc/len(y_true)
    sum_binary_acc = sum_binary_acc/len(y_true)
    metric_dict = {'Accuracy': acc, 'Binary Accuracy': sum_binary_acc}
    if metric_path is not None:
       json.dump(metric_dict,open(metric_path,'w'),indent=4)
    
    # print(acc)
    return metric_dict

def eval_performance(y_true, y_pred, metric_path=None,silent=False):

    # Precision
    metric_dict = {}

    # Accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    metric_dict['Accuracy'] = accuracy
    
    # Micro-F1 Score
    micro_f1 =  metrics.f1_score(y_true, y_pred, average='micro')
    metric_dict['Micro-F1'] = micro_f1

    # Macro-F1 Score
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    metric_dict['Macro-F1'] = macro_f1

    # Weighted-F1 Score
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    metric_dict['Weighted-F1'] = weighted_f1

    # Confusion matrix
    # print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))  
    if not silent:
        print("Accuracy:\n\t", accuracy)
        print("-------------------Micro-F1, Macro-F1, Weighted-F1..-------------------------")
        print("-------------------**********************************-------------------------")
        print("Micro-F1 Score:\n\t",micro_f1)
        print("Macro-F1 Score:\n\t", macro_f1)
        print("Weighted-F1 Score:\n\t", weighted_f1)
        print("-------------------**********************************-------------------------")


    if metric_path is not None:
       json.dump(metric_dict,open(metric_path,'w'),indent=4)

    return metric_dict