import os
import json
import argparse
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_dataset
from utils import create_gemini_config, request_gemini_engine, filter_none_python, filter_out_test_files, get_compressed_content

prompt = '''### Code Skeleton ###
{file_skeleton}

###

### Problem ###
{problem_statement}

###

Please provide a complete set of locations related to fixing the problem, including directly related ones as well as any potentially related functions, classes, methods and variables.
Please provide each location using one of the following formats and concatenate them with two blank lines.
```
Format 1
# file1.py
function: ...

Format 2
# file2.py
class: ...

Format 3
# file3.py
class: ...
method: ...

Format 4
# file4.py
variable: ...
```

Return the most important 30 locations.
Do not miss ``` before and after the locations.'''

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, default='results')
args = parser.parse_args()
if not os.path.isdir(os.path.join(args.pred_dir, 'func_localization')):
    os.makedirs(os.path.join(args.pred_dir, 'func_localization'))

def worker(e):
    d = json.load(open(os.path.join('repo_structures_verified', e["instance_id"] + ".json")))
    structure = d["structure"]
    filter_none_python(structure)
    if not d["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)
    with open(os.path.join(args.pred_dir, 'file_localization', e['instance_id']+'.json')) as f:
        preds = json.load(f)
    pred_files = preds['found_files'][:5]
    message = prompt.format(problem_statement=e['problem_statement'],
                            file_skeleton=get_compressed_content(pred_files, structure)).strip()
    config = create_gemini_config(message=message, temperature=0)
    exec_count = 0
    success = False
    ret = None
    while exec_count<3 and not success:
        exec_count += 1
        try:
            ret = request_gemini_engine(config,gemini_version='original')
            config['raw_output'] = ret['content'][0]['text']
            success = True
        except:
            pass
    if ret is None:
        return
    raw_locs = ret['content'][0]['text'].split('```')[1].split('\n\n')
    raw_locs = [l for l in raw_locs if l != '']
    locs = []
    for l in raw_locs:
        lines = l.strip().split('\n')
        locs.append({
            'file_name': lines[0].replace('#', '').strip(),
            'content': '\n'.join(lines[1:])
        })
    config['found_related_locs'] = locs
    with open(os.path.join(args.pred_dir, 'func_localization', e['instance_id'] + '.json'), 'w') as f:
        json.dump(config, f, indent=2)


swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

with mp.Pool(16) as pool, tqdm(total=len(swe_bench_data), desc='localize function') as pbar:
    for return_contents in pool.imap_unordered(worker, swe_bench_data):
        pbar.update()

# for e in tqdm(swe_bench_data):
#     worker(e)
