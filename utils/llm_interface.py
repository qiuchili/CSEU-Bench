import os
import time

from .const import (

    OPENAI_API_KEY,
    GEMINI_API_KEY,
    DASHSCOPE_API_KEY
)

def get_generation_func(model_name):

    if model_name.lower() == 'gemini-audio':
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0
            while retry_count < max_retries:
                try:
                    myfile = genai.upload_file(query[0]['audio_url'])
                    response = model.generate_content(
                        [myfile, query[1]['text']], 
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
            while retry_count < max_retries:
                try:
                    myfile = genai.upload_file(query[0]['audio_url'])
                    response = model.generate_content(
                        [myfile, query[1]['text']], 
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

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0
            while retry_count < max_retries:
                try:
                    myfile = genai.upload_file(query[0]['audio_url'])
                    response = model.generate_content(
                        [myfile, query[1]['text']], 
                        generation_config=genai.types.GenerationConfig(max_output_tokens=64)
                        )
                    time.sleep(3)
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
        client = OpenAI(api_key=OPENAI_API_KEY) 
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0

            while retry_count < max_retries:
                try:
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
                    completion = client.chat.completions.create(
                        model='gpt-4o-audio-preview',
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

    elif model_name.lower() == 'qwen-audio-turbo':
        from dashscope import MultiModalConversation
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY
        def gen(query):
            max_retries = 3
            delay = 2
            retry_count = 0
            while retry_count < max_retries:
                try:

                    result = MultiModalConversation.call(
                        model='qwen-audio-turbo-latest',
                        messages=[
                            {"role": "user",
                            "content": [
                                {"audio": query[0]['audio_url']},
                                {"text": query[1]['text']}
                                ]
                            }
                        ]
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
    
    