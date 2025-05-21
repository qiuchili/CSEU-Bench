import pandas as pd
import os
import argparse
import random
import numpy as np
import torch

from utils.const import TASK_LABELS_CSEU
from utils.generic import postprocess_llm_output, eval_performance_cseu
from utils.llm_interface import get_generation_func
from utils import audio_prompt_generator_fsl 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='语音大模型言语理解能力测试.')
    parser.add_argument('--task', metavar='T', type=str, help='task type', default='target_attitude')
    parser.add_argument('--features', metavar='F', type=str, help='feature list', default='t')
    parser.add_argument('--model_name', metavar='M', type=str, help='LLM name', default='qwen2-audio-instruct')
    parser.add_argument('--k', metavar='K', type=int, help='value of k for k-shot prompting', default=5)
    parser.add_argument('--datasets_dir', metavar='D', type=str, help='directory where CSEU-Bench locates', default='data')
    parser.add_argument('--dataset_name', metavar='N', type=str, help='name of the dataset file (.csv)', default='all.csv')
    parser.add_argument('--output_folder', metavar='O', type=str, help='result folder', default='output_cseu_fsl')
    parser.add_argument('--force_evaluate', help='whether or not evaluate performance', action='store_true',default=False)
    parser.add_argument('--seed', metavar='S', type=int, help='random seed', default=42)

    # Qwen2-Audio-7B-Instruct arguments
    parser.add_argument("--qwen2_audio_instruct_model_dir",type=str, default="pretrained_models/Qwen2-Audio-7B-Instruct")

    args = parser.parse_args()
    # args.force_evaluate = True
    # base_name = os.path.basename(args.dataset_path)
    print('task: {}; dataset: {}; features: {}; model: {}; seed: {}'.format(args.task, args.dataset_name, args.features, args.model_name, args.seed))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_output_dir = os.path.join(args.output_folder, args.model_name)
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)

    dataset_file_name = args.dataset_name.replace('.csv','')
    output_csv_file = os.path.join(model_output_dir, f'output_{args.task}_{args.features}_{dataset_file_name}_seed_{args.seed}.csv')
    eval_json_file =  os.path.join(model_output_dir,f'eval_{args.task}_{args.features}_{dataset_file_name}_seed_{args.seed}.json')

    all_labels = TASK_LABELS_CSEU[dataset_file_name][args.task]
    task_label_column = args.task

    if os.path.exists(output_csv_file):
        print('Experiment result already exists.')
    else:
        prompt_generate_func = getattr(audio_prompt_generator_fsl,f'{args.task}_{args.features}')
        llm = get_generation_func(args)
        df = pd.read_csv(os.path.join(args.datasets_dir, 'CSEU-Bench','processed_data', args.dataset_name),index_col=0)
        df = df[df[task_label_column].notna()]
        df['target_audio'] = args.datasets_dir +'/' + df['target_audio']
        df = df.reset_index(drop=True)
        labels = []
        prompts = []
        output_prompts = []
        for i, row in df.iterrows():
            
            print(f'Sample {i}/{len(df)}...')
            k_example_rows = df[df.index !=i].sample(args.k,replace=False,ignore_index=True)
            prompt = prompt_generate_func(row, k_example_rows, all_labels)
            result = llm(prompt)
            result, label = postprocess_llm_output(result,all_labels)
            labels.append(label)
        
            prompts.append(prompt)
            output_prompts.append(result)
            
        df['pred'] = labels
        df['input_prompt'] = prompts
        df['llm_output'] = output_prompts
        model_output_folder = os.path.join(args.output_folder, args.model_name)
        if not os.path.exists(model_output_folder):
            os.mkdir(model_output_folder)
        df.to_csv(output_csv_file)

    print("Evaluation....")
    if os.path.exists(eval_json_file) and not args.force_evaluate:
        print("Performance file already exists.")
    else: 
        df = pd.read_csv(output_csv_file,index_col=0)
        df = df[(df['llm_output'] != '') & df['llm_output'].notna()]
        df['pred'] = df['pred'].fillna('')
        # label_to_id_dict[''] = -1
        import pickle
        tree = pickle.load(open(os.path.join(args.datasets_dir, 'CSEU-Bench','processed_data/emotion_clustering_tree.pkl'), 'rb'))

        if dataset_file_name == 'discourse':
            # Sentimental/Neutral literal
            # Sentimental/Neutral audio
            # Neutral literal + Neutral voice
            literal_neutral_df = df[df['literal_sentiment'] == False]
            literal_sentimental_df = df[df['literal_sentiment'] == True]

            NN_df = literal_neutral_df[literal_neutral_df['target_audio'].str.endswith('N.wav')]
            NY_df = literal_neutral_df[~literal_neutral_df['target_audio'].str.endswith('N.wav')]
            YN_df = literal_sentimental_df[literal_sentimental_df['target_audio'].str.endswith('N.wav')]
            YY_df = literal_sentimental_df[~literal_sentimental_df['target_audio'].str.endswith('N.wav')]
            agree_df = pd.concat([NN_df, YY_df])
            disagree_df =  pd.concat([NY_df, YN_df])

            for max_depth in range(2, tree.depth()+1):
                eval_performance_cseu(NN_df[task_label_column].map(all_labels), 
                        NN_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}_NN.json'),max_depth=max_depth)
                eval_performance_cseu(NY_df[task_label_column].map(all_labels), 
                        NY_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}_NY.json'),max_depth=max_depth)
                eval_performance_cseu(YN_df[task_label_column].map(all_labels),   
                        YN_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}_YN.json'),max_depth=max_depth)
                eval_performance_cseu(YY_df[task_label_column].map(all_labels), 
                        YY_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}_YY.json'),max_depth=max_depth)
                eval_performance_cseu(agree_df[task_label_column].map(all_labels), 
                        agree_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}_agree.json'),max_depth=max_depth)
                eval_performance_cseu(disagree_df[task_label_column].map(all_labels), 
                        disagree_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}_disagree.json'),max_depth=max_depth)
        for max_depth in range(2, tree.depth()+1):
            eval_performance_cseu(df[task_label_column].map(all_labels), df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json',f'_depth_{max_depth}.json'),max_depth=max_depth)