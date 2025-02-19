# CSEU-Bench: Multi-faceted LLM Evaluation on Chinese Speech Emotional Understanding with Pyscho-linguistic Labels

**In compliance with data anonymization and GDPR policy, this repository only contains code for evaluating LLMs on a sample dataset for the moment. The full dataset will be released upon official acceptance of the paper**


## To run evaluation on CSEU-Bench:

**1. Set api keys or simply replace const.py**

**2. Replace the .csv files with real samples under /resource**

**3. Place all audio files under /audios**

**4. Run main_audio_csu.py for evaluating speech LLMs.**
- For SALMONN-13b: run `main_audio_csu_salmonn.py`
- For qwen2-audio-instruct: `run main_audio_csu_qwen2_instruct.py`
