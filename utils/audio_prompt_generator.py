from .generic import get_random_shuffled_labels_str
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