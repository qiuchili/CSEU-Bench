import pandas as pd
import os
import json
import numpy as np
from utils.label_clustering import get_distance
from utils.const import (
    ATTITUDE_LABELS_CSEU,
    ALL_TASKS_CSEU,
    ALL_MODELS_CSEU, 
)
import pickle
from ast import literal_eval
from utils.generic import (
    majority_vote
)
###################################################
# Utils for CSEU data dump
def parse_llm_eval_result_cseu(dir_path,max_depth=5):
    '''
    Read csv files of model predictions in the output folder
    And calculate the performances
    Return the original scores as well as the formatted score strings
    '''
    result_dict = dict()
    result_str_dict = dict()
    result_breakdown_dict = {'NN':dict(),'NY':dict(),'YN':dict(),'YY':dict(),'agree':dict(),'disagree':dict()}
    result_breakdown_str_dict = {'NN':dict(),'NY':dict(),'YN':dict(),'YY':dict(),'agree':dict(),'disagree':dict()}
    for model_name in os.listdir(dir_path):
        if not model_name in ALL_MODELS_CSEU:
            continue
        model_eval_dir = os.path.join(dir_path, model_name)

        # Stores all performance dicts
        # {seed: {task: [acc, f1], ...}, ...}
        model_performance_dicts = {}
        model_performance_breakdown_dicts = {'NN':dict(),'NY':dict(),'YN':dict(),'YY':dict(),'agree':dict(),'disagree':dict()}
        file_names = os.listdir(model_eval_dir)
        result_dict[model_name] = dict()
        result_str_dict[model_name] = dict()

        for s in ['NN','NY','YN','YY','agree','disagree']:
            result_breakdown_dict[s][model_name] = dict()
            result_breakdown_str_dict[s][model_name] = dict()

        for f in file_names:
            if not f.endswith('json'):
                continue
            if not f'depth_{max_depth}' in f:
                continue
            elif f.endswith('N.json') or f.endswith('Y.json') or f.endswith('agree.json'):
                info_segs = f.replace('.json','').split('_')
                breakdown_str = info_segs[-1]
                seed = int(info_segs[-4])

                # Read in the LLM predictions
                data = json.load(open(os.path.join(model_eval_dir, f)))
                dataset_name = info_segs[4]
                if not dataset_name in model_performance_breakdown_dicts[breakdown_str]:
                    model_performance_breakdown_dicts[breakdown_str][dataset_name] = {}
                model_performance_breakdown_dicts[breakdown_str][dataset_name][seed] = data['Accuracy']
            else:
                info_segs = f.replace('.json','').split('_')
                seed = int(info_segs[-3])

                # Read in the LLM predictions
                data = json.load(open(os.path.join(model_eval_dir, f)))
                dataset_name = info_segs[4]
                if not dataset_name in model_performance_dicts:
                    model_performance_dicts[dataset_name] = {}
                model_performance_dicts[dataset_name][seed] = data['Accuracy']

        for k in model_performance_dicts.keys():
            performance = model_performance_dicts[k]
            result_dict[model_name][k] = performance
            result_str_dict[model_name][k] = '${}^{{\\pm{}}}$'.format(
            round(np.mean(list(performance.values()))*100,2), round(np.std(list(performance.values()))*100,2))
        
        for breakdown_str, _model_performance_breakdown_dict in model_performance_breakdown_dicts.items():
            for k in _model_performance_breakdown_dict.keys():
                performance = _model_performance_breakdown_dict[k]
                result_breakdown_dict[breakdown_str][model_name][k] = performance
                result_breakdown_str_dict[breakdown_str][model_name][k] = '${}^{{\\pm{}}}$'.format(
                round(np.mean(list(performance.values()))*100,2), round(np.std(list(performance.values()))*100,2))
    return result_dict, result_str_dict, result_breakdown_dict, result_breakdown_str_dict

def get_average_human_performance(true_label, preds, LABEL_MAPPING_DICT, label_clustering_tree, max_depth=5):
    y_t = LABEL_MAPPING_DICT[true_label]
    sum_acc = 0
    for p in preds:
        if p not in LABEL_MAPPING_DICT:
            sum_acc += 0
        else:
            y_p = LABEL_MAPPING_DICT[p]
            distance = get_distance(y_t, y_p, label_clustering_tree,max_depth=max_depth)
            sum_acc += 1-distance

    sum_acc = sum_acc/len(preds)
    return sum_acc

def parse_human_eval_result_cseu(
        target_csv_file='data_pack/各种可用于评测的情感意图库/汇总0117.csv',
        label_cluster_tree_path='data_pack/情感相似度感知/human_perception_cluster.pkl',
        max_depth=5):

    tree = pickle.load(open(label_cluster_tree_path, 'rb'))
    df = pd.read_csv(target_csv_file, index_col=0)
    df['judgment'] = df['judgment'].apply(literal_eval)
    df['target_attitude'] = df['target_attitude'].replace({'bs':'ng','cq':'cf','nq':'ng','yq':'yw','gq':'gx','hq':'hp','sq':'sq','xf':'gx'})
    df['majority_acc'] = df.apply(lambda x: 1/len(majority_vote(x['judgment'])) if  x['target_attitude'] in majority_vote(x['judgment']) else 0, axis=1)
    # for max_depth in range(2, tree.depth()+1):

    df.loc[:,'mean_acc'] = df.apply(lambda x: get_average_human_performance(x['target_attitude'],x['judgment'],ATTITUDE_LABELS_CSEU,tree,max_depth=max_depth),axis=1)

    human_performance_dict = {}
    human_performance_str_dict = {}

    human_performance_breakdown_dict = {'NN':dict(),'NY':dict(),'YN':dict(),'YY':dict(),'agree':dict(),'disagree':dict()}
    human_performance_breakdown_str_dict = {'NN':dict(),'NY':dict(),'YN':dict(),'YY':dict(),'agree':dict(),'disagree':dict()}

    for sample_type in ALL_TASKS_CSEU:
        # data_df = pd.read_csv(posixpath.join('data_pack','各种可用于评测的情感意图库',file_name+'.csv'),index_col=0)

        data_df = df[df['sample_type'] == sample_type]
        
        # data_df['judgment'] = data_df['target_audio'].apply(lambda x: judgment_dict[x])
        human_performance_dict[sample_type] = dict()
        human_performance_str_dict[sample_type] = dict()
        human_performance_dict[sample_type] = data_df['mean_acc'].mean()
        human_performance_str_dict[sample_type] = '${}$'.format(round(data_df['mean_acc'].mean()*100,2))

        if sample_type == 'discourse':
            literal_neutral_df = data_df[data_df['literal_sentiment'] == False]
            literal_sentimental_df = data_df[data_df['literal_sentiment'] == True]

            NN_df = literal_neutral_df[literal_neutral_df['target_audio'].str.endswith('N.wav')]
            NY_df = literal_neutral_df[~literal_neutral_df['target_audio'].str.endswith('N.wav')]

            YN_df = literal_sentimental_df[literal_sentimental_df['target_audio'].str.endswith('N.wav')]
            YY_df = literal_sentimental_df[~literal_sentimental_df['target_audio'].str.endswith('N.wav')]
            agree_df = pd.concat([NN_df, YY_df])
            disagree_df =  pd.concat([NY_df, YN_df])

            df_breakdown_dict = {'NN':NN_df,'NY':NY_df,'YN':YN_df,'YY':YY_df,'agree':agree_df,'disagree':disagree_df}
            for name, _df in df_breakdown_dict.items():
                _df.loc[:,'mean_acc'] = _df.apply(lambda x: get_average_human_performance(x['target_attitude'],x['judgment'],ATTITUDE_LABELS_CSEU,tree,max_depth=max_depth),axis=1)

                human_performance_breakdown_dict[name][sample_type] = _df['mean_acc'].mean()
                human_performance_breakdown_str_dict[name][sample_type] = '${}$'.format(round(_df['mean_acc'].mean()*100,2))

    human_performance_dict['all'] = df['mean_acc'].mean()
    human_performance_str_dict['all'] = '${}$'.format(round(df['mean_acc'].mean()*100,2))

    return human_performance_dict, human_performance_str_dict, human_performance_breakdown_dict, human_performance_breakdown_str_dict
