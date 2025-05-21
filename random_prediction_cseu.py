import pandas as pd
import os
import argparse
import random
import numpy as np
import torch
from utils.const import TASK_LABELS_CSEU
from utils.generic import eval_performance_cseu
from utils import audio_prompt_generator 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='语音大模型言语理解能力测试.')
    parser.add_argument('--task', metavar='T', type=str, help='task type', default='target_attitude')
    parser.add_argument('--features', metavar='F', type=str, help='feature list', default='t')
    parser.add_argument('--datasets_dir', metavar='D', type=str, help='directory where CSEU-Bench locates', default='data')
    parser.add_argument('--dataset_name', metavar='N', type=str, help='name of the dataset file (.csv)', default='discourse.csv')
    parser.add_argument('--output_folder', metavar='O', type=str, help='result folder', default='cseu_output')
    parser.add_argument('--force_evaluate', help='whether or not evaluate performance', action='store_true',default=False)
    parser.add_argument('--seed', metavar='S', type=int, help='random seed', default=666)
    args = parser.parse_args()

    print('task: {}; dataset: {}; features: {}; model: {}; seed: {}'.format(args.task, args.dataset_name, args.features, 'random prediction', args.seed))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_output_dir = os.path.join(args.output_folder, 'random-prediction')
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
        prompt_generate_func = getattr(audio_prompt_generator,f'{args.task}_{args.features}')
        df = pd.read_csv(os.path.join(args.datasets_dir, 'CSEU-Bench', 'processed_data', args.dataset_name),index_col=0)
        df = df[df[task_label_column].notna()]
        df['target_audio'] = args.datasets_dir +'/' + df['target_audio']
        df = df.reset_index(drop=True)

        labels = []
        prompts = []
        output_prompts = []
        for i, row in df.iterrows():
            
            print(f'Sample {i}/{len(df)}...')
            label, output = random.choice(list(all_labels.items()))
            labels.append(label)
        
            prompts.append('Dummy prompt')
            output_prompts.append(output)
            
        df['pred'] =  labels
        df['input_prompt'] = prompts
        df['llm_output'] = output_prompts
        df.to_csv(output_csv_file)

    print("Evaluation....")
    if os.path.exists(eval_json_file) and not args.force_evaluate:
        print("Performance file already exists.")
    else: 
        df = pd.read_csv(output_csv_file,index_col=0)
        df = df[df['llm_output'] != '']
        df['pred'] = df['pred'].fillna('')
        import pickle
        tree = pickle.load(open(os.path.join(args.datasets_dir, 'processed_data/emotion_clustering_tree.pkl'), 'rb'))

        if dataset_file_name == 'discourse':
           
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