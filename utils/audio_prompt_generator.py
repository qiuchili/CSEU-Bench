from .generic import get_random_shuffled_labels_str

def context_implication_ct(row):
    all_label_str = get_random_shuffled_labels_str('context_implication')
    task_description = f"""
    ### 任务描述:
    下面是目标句语音产生的语境。请根据语境和目标句语音综合判断语境是表层还是深层。表层语境不包含言外意; 深层语境则表示冲突, 有言外意。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}
    
    ### 你的回答:
    """

    prompt = [
        {"type": "audio", "audio_url": row['target_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def target_sentiment_t(row):
    all_label_str = get_random_shuffled_labels_str('target_sentiment')
    task_description = f"""
    ### 任务描述:
    请判断该语音蕴含的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt
    
def target_sentiment_ct(row):
    all_label_str = get_random_shuffled_labels_str('target_sentiment')

    task_description = f"""
    ### 任务描述:
    下面是目标句语音产生的语境。请根据语境和目标句语音综合判断目标句蕴含的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def target_sentiment_tr(row):
    all_label_str = get_random_shuffled_labels_str('target_sentiment')


    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答。请根据语音综合判断目标句蕴含的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def target_sentiment_ctr(row):
    all_label_str = get_random_shuffled_labels_str('target_sentiment')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断目标句蕴含的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt
def target_attitude_t(row, labels=None):
    if labels is not None:
        all_label_str = get_random_shuffled_labels_str(labels)
    else:
        all_label_str = get_random_shuffled_labels_str('target_attitude')
    task_description = f"""
    ### 任务描述:
    请判断该语音蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt
    
def target_attitude_ct(row):
    all_label_str = get_random_shuffled_labels_str('target_attitude')
    task_description = f"""
    ### 任务描述:
    下面是目标句语音产生的语境。请根据语境和目标句语音综合判断目标句蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def target_attitude_tr(row):
    all_label_str = get_random_shuffled_labels_str('target_attitude')


    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答。请根据语音综合判断目标句蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def target_attitude_ctr(row):
    all_label_str = get_random_shuffled_labels_str('target_attitude')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断目标句蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

# def target_attitude_t(row):
#     all_label_str = get_random_sampled_labels_str('target_attitude', row['target_attitude'])
#     task_description = f"""
#     ### 任务描述:
#     请判断该语音蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 你的回答:
#     """
#     prompt = [
#         {"type": "audio", "audio_url": row['target_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt
    
# def target_attitude_ct(row):
#     all_label_str = get_random_sampled_labels_str('target_attitude', row['target_attitude'])
#     task_description = f"""
#     ### 任务描述:
#     下面是目标句语音产生的语境。请根据语境和目标句语音综合判断目标句蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 语境:
#     {row['context']}

#     ### 你的回答:
#     """
#     prompt = [
#         {"type": "audio", "audio_url": row['target_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt

# def target_attitude_tr(row):
#     all_label_str = get_random_sampled_labels_str('target_attitude', row['target_attitude'])


#     task_description = f"""
#     ### 任务描述:
#     输入语音为目标句以及对该句的回答。请根据语音综合判断目标句蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 你的回答:
#     """
#     prompt = [
#         {"type": "audio", "audio_url": row['target_response_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt

# def target_attitude_ctr(row):
#     all_label_str = get_random_sampled_labels_str('target_attitude', row['target_attitude'])

#     task_description = f"""
#     ### 任务描述:
#     输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断目标句蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 语境:
#     {row['context']}

#     ### 你的回答:
#     """
#     prompt = [
#         {"type": "audio", "audio_url": row['target_response_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt

def response_type_ctr(row):

    all_label_str = get_random_shuffled_labels_str('response_type')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断对目标句的回答是否为直接回答。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", 'audio_url': row['target_response_audio']},
        {"type": "text", 'text': task_description}
    ]
    return prompt

def response_type_tr(row):

    all_label_str = get_random_shuffled_labels_str('response_type')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答。请根据语音综合判断对目标句的回答是否为直接回答。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def response_sentiment_r(row):
    all_label_str = get_random_shuffled_labels_str('response_sentiment')

    task_description = f"""
    ### 任务描述:
    请判断该语音蕴含的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def response_sentiment_tr(row):
    all_label_str = get_random_shuffled_labels_str('response_sentiment')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答。请根据语音综合判断对目标句的回答的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def response_sentiment_ctr(row):
    all_label_str = get_random_shuffled_labels_str('response_sentiment')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断对目标句的回答的情感。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
   
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def response_attitude_r(row):
    all_label_str = get_random_shuffled_labels_str('response_attitude')

    task_description = f"""
    ### 任务描述:
    请判断该语音蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def response_attitude_tr(row):
    all_label_str = get_random_shuffled_labels_str('response_attitude')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答。请根据语音综合判断对目标句的回答的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 你的回答:
    """
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

def response_attitude_ctr(row):
    all_label_str = get_random_shuffled_labels_str('response_attitude')

    task_description = f"""
    ### 任务描述:
    输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断对目标句的回答的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

    ### 语境:
    {row['context']}

    ### 你的回答:
    """
   
    prompt = [
        {"type": "audio", "audio_url": row['target_response_audio']},
        {"type": "text", "text": task_description},
    ]
    return prompt

# def response_attitude_r(row):
#     all_label_str = get_random_sampled_labels_str('response_attitude', row['response_attitude'])

#     task_description = f"""
#     ### 任务描述:
#     请判断该语音蕴含的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 你的回答:
#     """
#     prompt = [
#         {"type": "audio", "audio_url": row['response_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt

# def response_attitude_tr(row):
#     all_label_str = get_random_sampled_labels_str('response_attitude', row['response_attitude'])

#     task_description = f"""
#     ### 任务描述:
#     输入语音为目标句以及对该句的回答。请根据语音综合判断对目标句的回答的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 你的回答:
#     """
#     prompt = [
#         {"type": "audio", "audio_url": row['target_response_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt

# def response_attitude_ctr(row):
#     all_label_str = get_random_sampled_labels_str('response_attitude', row['response_attitude'])

#     task_description = f"""
#     ### 任务描述:
#     输入语音为目标句以及对该句的回答，该段对话发生的语境如下。请根据语境，目标句和对目标句的回答，综合判断对目标句的回答的细粒度情感态度。请在 {all_label_str} 中选择最合适的选项。

#     ### 语境:
#     {row['context']}

#     ### 你的回答:
#     """
   
#     prompt = [
#         {"type": "audio", "audio_url": row['target_response_audio']},
#         {"type": "text", "text": task_description},
#     ]
#     return prompt

