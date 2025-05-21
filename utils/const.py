OPENAI_API_KEY='your_api_key'
WANDB_API_KEY='your_api_key'
HF_API_KEY='your_api_key'
DEEPSEEK_API_KEY='your_api_key'
ANTHROPIC_API_KEY='your_api_key'
MOONSHOT_API_KEY='your_api_key'
GEMINI_API_KEY='your_api_key'
DASHSCOPE_API_KEY='your_api_key'

ALL_DATASETS_CSEU = ['1-七种情感库1男1女','2-相同文本的几种情绪-字面有无情绪','2-相同文本的几种情绪-第一部分','2-相同文本的几种情绪-第二部分', '4-情绪态度库']
ALL_TASKS_CSEU = ['single-syllable','two-syllable','short-sentence','discourse']
ALL_MODELS_CSEU = ['random-prediction','salmonn-13b','kimi-audio-7b-instruct','qwen2-audio-instruct','qwen-audio-turbo', 'gemini-audio','gemini-pro-audio','gemini-2.0-flash-audio','gpt-4o-audio','human']
MODEL_COLOR_DICT_CSEU = {'random-prediction':'black',
                         'salmonn-13b':'lightcyan',
                         'kimi-audio-7b-instruct':'lightcoral',
                         'qwen2-audio-instruct':'moccasin',
                         'qwen-audio-turbo':'lightseagreen', 
                         'gemini-audio':'lightpink',
                         'gemini-pro-audio':'lightblue',
                         'gemini-2.0-flash-audio':'lightgreen',
                         'gpt-4o-audio':'lightyellow',
                         'human':'indigo'}

ATTITUDE_LABELS_CSEU = {
        'gx':'高兴','ganx':'感谢/感激','ja':'骄傲/荣耀','jid':'激动',
        'kw':'渴望/盼望','my':'满意','mz':'满足','qq':'祈求',
        'qx':'庆幸','qxu':'谦虚/谦卑','rq':'热情/热心','xa':'喜爱',
        'xx':'相信/信任','xm':'羡慕','yh':'友好','zj':'尊敬',
        'zy':'赞扬/表扬','hx':'害羞','fn':'愤怒','sq':'生气',
        'ng':'难过','hp':'害怕/恐惧','yw':'厌恶','bq':'抱歉/道歉',
        'by':'抱怨','cf':'嘲讽/讽刺','dx':'担心','fm':'烦闷/愁闷',
        'gg':'尴尬','gj':'告诫/劝告','hh':'后悔','jd':'嫉妒',
        'jj':'焦急/急躁','juj':'拒绝/谢绝','jz':'紧张','kh':'恐吓/吓唬',
        'lm':'怜悯/可怜','ns':'难受','qs':'轻视/蔑视','sb':'申辩/狡辩',
        'sw':'失望','tf':'颓废/倦怠','th':'讨好','tx':'挑衅',
        'wn':'无奈','wq':'委屈','wx':'惋惜','xk':'羞愧/愧疚',
        'xy':'炫耀','yih':'遗憾','yc':'忧愁','yuh':'怨恨',
        'za':'自傲','zg':'责怪/指责','ziz':'自责','fb':'反驳',
        'pp':'批评','ml':'命令', 'jx':'惊喜','aw':'安慰',
        'jy':'惊讶','cc':'猜测/猜忌','gk':'感慨','pj':'平静/宁静',
        'xw':'醒悟','yihuo':'疑惑','yy':'犹豫','wenx':'问询',
        'jyi':'建议','xn':'许诺','wh':'维护/解释', 
        'qd':'强调','gd':'感动','js':'谨慎',
        'hq':'好奇','tk':'调侃','lmo':'冷漠',
        'fy':'敷衍','hy':'怀疑/质疑', 'bnf':'不耐烦',
        'jdi':'坚定','juej':'倔强','zx':'中性'}

TASK_LABELS_CSEU = {
    'single-syllable':{
        
        'target_attitude': ATTITUDE_LABELS_CSEU
    },

    'two-syllable':{
        'target_attitude': ATTITUDE_LABELS_CSEU
    },

    'short-sentence':{

        'target_attitude': ATTITUDE_LABELS_CSEU
    },

    'discourse':{
        'target_attitude': ATTITUDE_LABELS_CSEU
    },
    'all':{
        'target_attitude': ATTITUDE_LABELS_CSEU
    },
    
}