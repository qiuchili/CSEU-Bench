import pandas as pd
import os
import argparse
import random
import numpy as np
import torch

from utils.const import TASK_LABELS_CSU
from utils.generic import postprocess_llm_output, eval_performance
from utils import audio_prompt_generator 
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='语音大模型言语理解能力测试.')
    parser.add_argument("--model_dir",type=str, default="/disks/sdb/model_space/Qwen2-Audio-7B-Instruct")
    parser.add_argument('--task', metavar='T', type=str, help='task type', default='target_attitude')
    parser.add_argument('--features', metavar='F', type=str, help='feature list', default='t')
    # parser.add_argument('--model_name', metavar='M', type=str, help='LLM name', default='qwen2-audio-instruct')
    parser.add_argument('--dataset_path', metavar='D', type=str, help='dataset path', default='resource/single-syllable.csv')
    parser.add_argument('--output_folder', metavar='O', type=str, help='result folder', default='output_csu')
    parser.add_argument('--seed', metavar='S', type=int, help='random seed', default=666)
    parser.add_argument('--force_evaluate', help='whether or not evaluate performance', action='store_true',default=False)

    args = parser.parse_args()
    args.model_name = 'qwen2-audio-instruct'
    
    base_name = os.path.basename(args.dataset_path)

    print('task: {}; dataset: {}; features: {}; model: {}; seed: {}'.format(args.task, base_name, args.features, args.model_name, args.seed))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_output_dir = os.path.join(args.output_folder, args.model_name)
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)

    dataset_file_name = base_name.replace('.csv','')
    output_csv_file = os.path.join(model_output_dir, f'output_{args.task}_{args.features}_{dataset_file_name}_seed_{args.seed}.csv')
    eval_json_file =  os.path.join(model_output_dir,f'eval_{args.task}_{args.features}_{dataset_file_name}_seed_{args.seed}.json')
    task_label_column = args.task
    all_labels = TASK_LABELS_CSU[dataset_file_name][args.task]
    if os.path.exists(output_csv_file):
        print('Experiment result already exists.')
    
    else:
        prompt_generate_func = getattr(audio_prompt_generator,f'{args.task}_{args.features}')
        processor = AutoProcessor.from_pretrained(args.model_dir)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_dir,device_map='auto')
        df = pd.read_csv(args.dataset_path)
        
        df = df[df[task_label_column].notna()]
        df = df.reset_index(drop=True)
        label_to_id_dict = {k: i for i, (k, v) in enumerate(all_labels.items())}

        labels = []
        prompts = []
        output_prompts = []
        for i, row in df.iterrows():
            
            print(f'Sample {i}/{len(df)}...')
            prompt = prompt_generate_func(row,all_labels)
            conversation = [
                {"role": "user", "content": prompt}
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(librosa.load(
                                ele['audio_url'], 
                                sr=processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = processor(text=text, audios=audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs.feature_attention_mask = inputs.feature_attention_mask.to('cuda')
            inputs.attention_mask = inputs.attention_mask.to('cuda')
            inputs.input_ids = inputs.input_ids.to('cuda')
            inputs.input_features = inputs.input_features.to('cuda')
            generate_ids = model.generate(**inputs, max_new_tokens=128)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result, label = postprocess_llm_output(result,all_labels)
            labels.append(label)
        
            prompts.append(prompt)
            output_prompts.append(result)
            
        df['pred'] =  labels
        df['input_prompt'] = prompts
        df['llm_output'] = output_prompts
        model_output_folder = os.path.join(args.output_folder, args.model_name)
        if not os.path.exists(model_output_folder):
            os.mkdir(model_output_folder)
        df.to_csv(output_csv_file,index=None)

        print("Evaluation....")
    if os.path.exists(output_csv_file) and not args.force_evaluate:
        print("Performance file already exists.")
    else: 
        df = pd.read_csv(output_csv_file,index_col=0)
        df = df[df['llm_output'] != '']
        df['pred'] = df['pred'].fillna('')
        import pickle
        tree = pickle.load(open('resource/human_perception_cluster.pkl', 'rb'))

        if dataset_file_name == 'discourse':
            # Sentimental/Neutral literal
            # Sentimental/Neutral audio
            main_data_path = 'resource/CESU_samples.csv'
            main_data_df = pd.read_csv(main_data_path,index_col=0)

            # Neutral literal + Neutral voice
            literal_neutral_df = df.loc[main_data_df[(main_data_df['literal_sentiment'] == False)].index.intersection(df.index),:]
            literal_sentimental_df = df.loc[main_data_df[(main_data_df['literal_sentiment'] == True)].index.intersection(df.index),:]

            NN_df = literal_neutral_df[literal_neutral_df['target_audio'].str.endswith('N.wav')]
            NY_df = literal_neutral_df[~literal_neutral_df['target_audio'].str.endswith('N.wav')]

            YN_df = literal_sentimental_df[literal_sentimental_df['target_audio'].str.endswith('N.wav')]
            YY_df = literal_sentimental_df[~literal_sentimental_df['target_audio'].str.endswith('N.wav')]

            eval_performance(NN_df[task_label_column].map(all_labels), 
                    NN_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json','_NN.json'))
            eval_performance(NY_df[task_label_column].map(all_labels), 
                    NY_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json','_NY.json'))
            eval_performance(YN_df[task_label_column].map(all_labels),   
                    YN_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json','_YN.json'))
            eval_performance(YY_df[task_label_column].map(all_labels), 
                    YY_df['pred'].map(all_labels), tree, metric_path=eval_json_file.replace('.json','_YY.json'))

        eval_performance(df[task_label_column].map(all_labels), 
                    df['pred'].map(all_labels), tree, metric_path=eval_json_file)