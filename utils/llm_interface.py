from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from huggingface_hub import get_inference_endpoint
from huggingface_hub import InferenceClient
import time
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from .const import (

    GEMINI_API_KEY,
    DASHSCOPE_API_KEY
)

def get_generation_func(args):
    model_name = args.model_name
    ###########################################################################
    if model_name.lower() == 'gemini-audio':
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            if isinstance(query[0], str):
                with open(query[0]['audio_url'], "rb") as f:
                    audio_data = f.read()
                prompt_parts = [
                    {"role": "user", "parts": [
                        {"mime_type": "audio/wav", "data": audio_data},
                        {"text": query[1]['text']}
                    ]}
                ]
            else:
                prompt_parts = []
                for q in query:
                    with open(q[0]['audio_url'], "rb") as f:
                        audio_data = f.read()
                    p = {"role": "user", "parts": [
                            {"mime_type": "audio/wav", "data": audio_data},
                            {"text": q[1]['text']}
                        ]}
                    prompt_parts.append(p)

            while retry_count < max_retries:
                try:
                    response = model.generate_content(
                        contents=prompt_parts, 
                        generation_config=genai.types.GenerationConfig(max_output_tokens=64)
                        )
                    return response.text
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
            return ""         
        return gen

    elif model_name.lower() == 'gemini-pro-audio':
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")

        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            if isinstance(query[0], str):
                with open(query[0]['audio_url'], "rb") as f:
                    audio_data = f.read()
                prompt_parts = [
                    {"role": "user", "parts": [
                        {"mime_type": "audio/wav", "data": audio_data},
                        {"text": query[1]['text']}
                    ]}
                ]
            else:
                prompt_parts = []
                for q in query:
                    with open(q[0]['audio_url'], "rb") as f:
                        audio_data = f.read()
                    p = {"role": "user", "parts": [
                            {"mime_type": "audio/wav", "data": audio_data},
                            {"text": q[1]['text']}
                        ]}
                    prompt_parts.append(p)

            while retry_count < max_retries:
                try:
                    response = model.generate_content(
                        contents=prompt_parts, 
                        generation_config=genai.types.GenerationConfig(max_output_tokens=128)
                        )
                    return response.text
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
            return ""         
        return gen

    elif model_name.lower() == 'gemini-2.0-flash-audio':
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-001")

        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            if isinstance(query[0], str):
                with open(query[0]['audio_url'], "rb") as f:
                    audio_data = f.read()
                prompt_parts = [
                    {"role": "user", "parts": [
                        {"mime_type": "audio/wav", "data": audio_data},
                        {"text": query[1]['text']}
                    ]}
                ]
            else:
                prompt_parts = []
                for q in query:
                    with open(q[0]['audio_url'], "rb") as f:
                        audio_data = f.read()
                    p = {"role": "user", "parts": [
                            {"mime_type": "audio/wav", "data": audio_data},
                            {"text": q[1]['text']}
                        ]}
                    prompt_parts.append(p)

            while retry_count < max_retries:
                try:
                    response = model.generate_content(
                        contents=prompt_parts, 
                        generation_config=genai.types.GenerationConfig(max_output_tokens=128)
                        )
                    return response.text
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
            return ""         
        return gen
 
    elif model_name.lower() == 'gpt-4o-audio':    
        from openai import OpenAI
        import base64
        client = OpenAI() 
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # Zero-shot
                    if isinstance(query[0], str):
                        with open(query[0]['audio_url'], "rb") as f:
                            audio_data = f.read()
                        encoded_string = base64.b64encode(audio_data).decode('utf-8')
                        
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    query[1],
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": encoded_string,
                                            "format": "wav"
                                        }
                                    }
                                ]
                            }
                        ]
                    else:
                        messages = []
                        for q in query:
                            with open(q[0]['audio_url'], "rb") as f:
                                audio_data = f.read()
                            encoded_string = base64.b64encode(audio_data).decode('utf-8')
                            
                            m = {"role": "user",
                                    "content": [
                                        q[1],
                                        {
                                            "type": "input_audio",
                                            "input_audio": {
                                                "data": encoded_string,
                                                "format": "wav"
                                            }
                                        }
                                    ]
                                }
                                    
                            messages.append(m)
                  
                    completion = client.chat.completions.create(
                        model='gpt-4o-audio-preview',
                        audio={"voice": "echo", "format": "wav"},
                        messages=messages,
                        max_tokens=100
                    )

                    msg = None
                    choices = completion.choices
                    if choices:
                        msg = choices[0].message.content
                    else:
                        msg = completion.message.content
                    return msg
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
            return ""      
        return gen

    elif model_name.lower() == 'qwen2-audio-instruct':

        import librosa
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.qwen2_audio_instruct_model_dir)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.qwen2_audio_instruct_model_dir,device_map='auto')
        model.eval()
        
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            # Zero-shot
            while retry_count < max_retries:
                try:
                    if isinstance(query[0], str):
                        conversation = [
                            {"role": "user", "content": query}
                        ]
                    else:
                        conversation = []
                        for q in query:
                            message = {"role": "user", "content": [
                                {"type": "audio", "audio_url": q[0]['audio_url']},
                                {"type": "text", "text": q[1]['text']}]
                            }
                            conversation.append(message) 
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios = []
                    for message in conversation:
                        if isinstance(message["content"], list):
                            for ele in message["content"]:
                                if ele["type"] == "audio":
                                    audios.append(librosa.load(
                                        ele['audio_url'], 
                                        sr=processor.feature_extractor.sampling_rate)[0])

                    inputs = processor(text=text, audios=audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
                    inputs.feature_attention_mask = inputs.feature_attention_mask.to('cuda')
                    inputs.attention_mask = inputs.attention_mask.to('cuda')
                    inputs.input_ids = inputs.input_ids.to('cuda')
                    inputs.input_features = inputs.input_features.to('cuda')
                    generate_ids = model.generate(**inputs, max_new_tokens=128)
                    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

                    result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    time.sleep(1)
                    return result
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
                return ""  
        return gen
    
    elif model_name.lower() == 'qwen-audio-turbo':
        from dashscope import MultiModalConversation
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            # Zero-shot
            if isinstance(query[0], str):
                messages=[
                            {"role": "user", "content": [
                                {"audio": query[0]['audio_url']},
                                {"text": query[1]['text']}]
                            }
                        ]
            else:
                messages = []
                for q in query:
                    m = {"role": "user",
                            "content": [
                                {"audio": q[0]['audio_url']},
                                {"text": q[1]['text']}
                                ]
                            }
                    messages.append(m)

            while retry_count < max_retries:
                try:

                    result = MultiModalConversation.call(
                        #model='qwen-audio-turbo-latest',
                        model='qwen-audio-turbo-2024-12-04',
                        messages=messages
                    )
                    result = result.output.choices[0].message['content'][0]['text']
                    time.sleep(1)
                    return result
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
                return ""  
        return gen
    
    elif model_name.lower() == 'salmonn-13b':
        from utils.salmonn.model import SALMONN

        model = SALMONN(
            ckpt=os.path.join(args.salmonn_model_dir,args.salmonn_ckpt_path),
            whisper_path=os.path.join(args.salmonn_model_dir,args.salmonn_whisper_path),
            beats_path=os.path.join(args.salmonn_model_dir,args.salmonn_beats_path),
            vicuna_path=os.path.join(args.salmonn_model_dir,args.salmonn_vicuna_path),
            low_resource=args.salmonn_low_resource,
            lora_alpha=28,
        )
        model.to(args.salmonn_device)
        model.eval()

        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0
            while retry_count < max_retries:
                try:

                    result = model.generate(
                        wav_path=query[0]['audio_url'],
                        prompt=query[1]['text']
                    )
                    time.sleep(1)
                    return result[0]
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
                return ""  
        return gen
    elif model_name.lower() == 'kimi-audio-instruct':
        from utils.kimia_infer.api.kimia import KimiAudio
        model = KimiAudio(
            model_path=args.kimi_audio_instruct_model_dir,
            load_detokenizer=False,
        )
        sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
            "max_new_tokens": 64
        }
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0
            while retry_count < max_retries:
                try:
                    messages = [
                            {"role": "user", "message_type": "text", "content": query[1]['text']},
                            {
                                "role": "user",
                                "message_type": "audio",
                                "content": query[0]['audio_url'],
                            },
                        ]
                        
                    _, result = model.generate(messages, **sampling_params, output_type="text")
                    return result[0]
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1       
                return ""  
        return gen