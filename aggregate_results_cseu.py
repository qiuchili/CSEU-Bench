
import pandas as pd
import os
from utils.performance_aggregator import (
        parse_llm_eval_result_cseu, 
        parse_human_eval_result_cseu
    )
from utils.generic import reorder_performance_table_cseu
from utils.latex import create_latex_performance_table_cseu
from utils.draw_plots import draw_radar_map

def main(llm_output_dir='output_cseu',
        dataset_folder = 'data/CSEU-Bench',
        output_dir = 'summary_cseu',
        max_depth=5):
    
    dataset_split_rename_map = {'single-syllable':'CSEU-monosyllabic','two-syllable':'CSEU-bisyllabic','short-sentence':'CSEU-short-sentence','discourse':'CSEU-discourse','all':'CSEU-all'}

    all_samples_file = os.path.join(dataset_folder,'processed_data/all_data.csv')
    emotion_cluster_file = os.path.join(dataset_folder,'processed_data/emotion_clustering_tree.pkl')
    output_csv_path = os.path.join(output_dir,f'performance_all_depth_{max_depth}.csv')
    human_dict, human_str_dict, human_breakdown_dict, human_breakdown_str_dict = parse_human_eval_result_cseu(all_samples_file, emotion_cluster_file,max_depth=max_depth)

    result_dict, result_str_dict, result_breakdown_dict, result_breakdown_str_dict = parse_llm_eval_result_cseu(llm_output_dir,max_depth=max_depth)
    result_str_dict['human'] = human_str_dict
    
    if max_depth == 5:
        result_dict['human'] = human_dict

        result_dict_for_radar_map = dict()
        all_labels = list(dataset_split_rename_map.keys())
        # print(all_labels)
        
        for k,v in result_dict.items():
            acc_values = []
            for l in all_labels:
                if l not in v:
                    val = -1
                else:
                    if isinstance(v[l],dict):
                        val = sum(v[l].values())/len(v[l].values())
                    else:
                        val = v[l]
                acc_values.append(val)
            result_dict_for_radar_map[k] = acc_values
        all_labels = [dataset_split_rename_map[l] if l in dataset_split_rename_map else l for l in all_labels]
        
        data_for_radar_map = [all_labels, result_dict_for_radar_map]
        
        draw_radar_map(data_for_radar_map, output_path=os.path.join(output_dir,'radar_map.png'))
    df = pd.DataFrame.from_dict(result_str_dict).T
    df = reorder_performance_table_cseu(df)
    df = df.rename(columns=dataset_split_rename_map)
    df = df.T
    df.to_csv(output_csv_path)

    latex_str = create_latex_performance_table_cseu(df)
    with open(os.path.join(output_dir,f'latex_performance_table_depth_{max_depth}.txt'),'w',encoding='utf-8') as f:
        print(latex_str,file=f)

    result_breakdown_dfs = []
    for breakdown_str in result_breakdown_str_dict:
        result_breakdown_str_dict[breakdown_str]['human'] = human_breakdown_str_dict[breakdown_str]

        breakdown_df = pd.DataFrame.from_dict(result_breakdown_str_dict[breakdown_str]).T
        breakdown_df = reorder_performance_table_cseu(breakdown_df)
        breakdown_df = breakdown_df.T
        breakdown_df.index= [breakdown_str]
        result_breakdown_dfs.append(breakdown_df)

    result_breakdown_df = pd.concat(result_breakdown_dfs)
    result_breakdown_df.to_csv(output_csv_path.replace('.csv',f'_breakdown.csv'))
    latex_str = create_latex_performance_table_cseu(result_breakdown_df)
    with open(os.path.join(output_dir,f'latex_performance_breakdown_table_depth_{max_depth}.txt'),'w',encoding='utf-8') as f:
        print(latex_str,file=f)

if __name__ == '__main__':
    llm_output_dir='output_cseu_fsl'
    dataset_folder ='data/CSEU-Bench'
    output_dir = 'summary_cseu_fsl'
    for max_depth in range(2, 11):
        main(llm_output_dir, dataset_folder, output_dir,max_depth=max_depth)

    llm_output_dir='output_cseu'
    dataset_folder ='data/CSEU-Bench'
    output_dir = 'summary_cseu'
    for max_depth in range(2, 11):
        main(llm_output_dir, dataset_folder, output_dir,max_depth=max_depth)
