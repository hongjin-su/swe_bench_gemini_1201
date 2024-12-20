import os
import json
import argparse
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_dataset
from collections import Counter
from utils import create_gemini_config, request_gemini_engine

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, default='results')
parser.add_argument("--gemini_version", type=str)
args = parser.parse_args()
if not os.path.isdir(os.path.join(args.pred_dir, 'repairs')):
    os.makedirs(os.path.join(args.pred_dir, 'repairs'))

def parse_output(s):
    components = s.split('```python')
    replacements = []
    for c in components:
        replacements.append(c.split('```')[0])
    return replacements

repair_prompt = '''### Code ###
{code_content}

###

### Problem ###
{problem_statement}

###

Generate the most important edit using the following format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

For example:
```python
### xxx/xx.py
<<<<<<< SEARCH
original code
=======
code after modification
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Don't forget import.
Wrap each *SEARCH/REPLACE* edit in blocks ```python...```.'''

def worker(e):
    if os.path.isfile(os.path.join(args.pred_dir, 'repairs', e['instance_id'] + '.json')):
        return
    d = json.load(open(os.path.join('repo_structures_verified', e["instance_id"] + ".json")))
    structure = d["structure"]
    if not os.path.isfile(os.path.join(args.pred_dir, 'func_localization', e['instance_id'] + '.json')):
        return
    with open(os.path.join(args.pred_dir, 'func_localization', e['instance_id'] + '.json')) as f:
        config = json.load(f)
    locs = config['found_related_locs']

    all_predictions = {}
    all_lines = []
    selected_file_contents = {}
    maximum_count = 0
    for loc in locs:
        pred_file_name = loc['file_name']
        if not pred_file_name.endswith('.py'):
            continue
        if not pred_file_name in all_predictions:
            all_predictions[pred_file_name] = []
        predictions = []
        directories = pred_file_name.split('/')
        file_contents = structure
        try:
            for d in directories:
                file_contents = file_contents[d]
        except Exception as error:
            continue
        assert not pred_file_name in selected_file_contents or selected_file_contents[pred_file_name]==file_contents
        if not pred_file_name in selected_file_contents:
            selected_file_contents[pred_file_name] = file_contents
        lines = loc['content'].split('\n')
        cur_class_name = None
        for idx in range(len(lines)):
            if lines[idx].startswith('function:'):
                cur_func = lines[idx].replace('function:','').strip()
                for func in file_contents['functions']:
                    if func['name']==cur_func:
                        predictions.append([func['start_line'], func['end_line']])
            elif lines[idx].startswith('variable:'):
                cur_var = lines[idx].replace('variable:','').strip()
                for line_idx,text_line in enumerate(file_contents['text']):
                    if cur_var+'=' in text_line or cur_var+' =' in text_line:
                        predictions.append([line_idx+1, line_idx+3])
            elif lines[idx].startswith('class:') and (idx==len(lines)-1 or not lines[idx+1].startswith('method')):
                cur_class = lines[idx].replace('class:','').strip()
                for c in file_contents['classes']:
                    if c['name']==cur_class:
                        predictions.append([c['start_line'], c['end_line']])
            elif lines[idx].startswith('method:'):
                pred_method = lines[idx].replace('method:','').strip()
                cur_methods = None
                for c in file_contents['classes']:
                    assert cur_methods is None
                    if c['name']==cur_class_name:
                        cur_methods = c['methods']
                        break
                if cur_methods is not None:
                    for m in cur_methods:
                        if pred_method == m['name']:
                            predictions.append([m['start_line'], m['end_line']])
            elif lines[idx].startswith('class:'):
                cur_class_name = lines[idx].replace('class:','').strip()
        all_predictions[pred_file_name] += predictions
        if len(predictions)>0:
            maximum_count += 1
        if maximum_count>5:
            break
    new_predictions_all = {}
    for file_name_index,content_predictions in all_predictions.items():
        all_lines.append('\n#'+file_name_index)
        raw_predictions = sorted(content_predictions,key=lambda x:x[0])
        predictions = []
        for p in raw_predictions:
            skip_flag = False
            for p1 in predictions:
                if p[0]>=p1[0] and p[1]<=p1[1]:
                    skip_flag = True
            if not skip_flag:
                predictions.append(p)
        for p in predictions:
            all_lines += ['\n']+selected_file_contents[file_name_index]['text'][p[0]-1: p[1]]
        new_predictions_all[file_name_index] = predictions
    code_content = '\n'.join(all_lines)
    repair_message = repair_prompt.format(code_content=code_content,problem_statement=e['problem_statement'])
    config = create_gemini_config(message=repair_message, temperature=0.8)

    config['raw_output'] = []
    for _ in range(1):
        ret = request_gemini_engine(config,gemini_version=args.gemini_version)
        if isinstance(ret,dict):
            config['raw_output'].append(ret['content'][0]['text'])
        else:
            config['raw_output'].append('')
    count = Counter(config['raw_output'])
    count = sorted(count.items(),key=lambda x:x[1],reverse=True)
    config['raw_output'] = []
    for modification,c in count:
        config['raw_output'].append(modification)

    with open(os.path.join(args.pred_dir, 'repairs', e['instance_id'] + '.json'), 'w') as f:
        json.dump(config, f, indent=2)


swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
swe_bench_data = [e for e in swe_bench_data]

with mp.Pool(16) as pool, tqdm(total=len(swe_bench_data), desc='edit') as pbar:
    for return_contents in pool.imap_unordered(worker, swe_bench_data):
        pbar.update()

# for e in tqdm(swe_bench_data[:10]):
#     worker(e)