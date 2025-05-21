import re
import numpy as np

def create_latex_performance_table_cseu(df):
    latex_str = df.to_latex(na_rep='')
    latex_to_paste = latex_str.split('\\midrule')[1].split('\\bottomrule')[0].strip()

    metric_lines = latex_to_paste.strip().split('\\\\')[:-1]

    new_metric_lines = []
    for _, one_line in enumerate(metric_lines):
        one_line = one_line.strip()
        items = one_line.split(' & ')
        task_name = items[0]
        new_items = [task_name]
        all_performances = []
        for item in items[1:]:
            s = re.findall(r'(?<=\$)(.*?)(?=\^|\$)',item)
            if len(s) > 0:
                s = float(s[0])
            else:
                s = -1
            all_performances.append(s)
        array = np.asarray(all_performances)
        max_index = np.argmax(array)

        to_be_replaced_content = items[max_index+1].split('$')[1]
        items[max_index+1] = items[max_index+1].replace(f'${to_be_replaced_content}$',f'\\bm{{${to_be_replaced_content}$}}')

        new_items.extend(items[1:])
        new_metric_line = ' & '.join(new_items) + ' \\\\ \hline' 
            
        new_metric_lines.append(new_metric_line)
    return '\n'.join(new_metric_lines)