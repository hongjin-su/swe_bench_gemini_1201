import os
import json
import random
import argparse
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_dataset
from utils import create_gemini_config, request_gemini_engine, filter_none_python, filter_out_test_files, show_project_structure

prompt = """### Repository Structure ###
{structure}

###

### GitHub Problem Description ###
{problem_statement}

###

Please look through the Repository structure and GitHub problem description and provide a list of files that one would need to edit to fix the problem.
Please only provide the full path and return the most important 20 files.
The returned files should be separated by new lines and wrapped with ```
For example:
```
django/db/models/fields/__init__.py
django/db/migrations/serializer.py
django/conf/__init__.py
sympy/core/mul.py
...
```

Do not miss ``` before and after the files.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, default='results')
args = parser.parse_args()
args.pred_dir = os.path.join(args.pred_dir, 'file_localization')
if not os.path.isdir(args.pred_dir):
    os.makedirs(args.pred_dir)


def worker(e):
    if os.path.isfile(os.path.join(args.pred_dir, e['instance_id'] + '.json')):
        return
    d = json.load(open(os.path.join('repo_structures_verified', e["instance_id"] + ".json")))
    structure = d["structure"]
    filter_none_python(structure)
    if not d["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)

    repo_structure = show_project_structure(structure).strip()
    message1 = prompt.format(problem_statement=e['problem_statement'],
                             structure=repo_structure).strip()
    config = create_gemini_config(message=message1, temperature=0.8)
    config['instance_id'] = e['instance_id']
    ret1 = request_gemini_engine(config,gemini_version='original')
    found_files = ret1['content'][0]['text'].split('```')[1].split('\n')
    raw_files = [file for file in found_files if file != '']
    config['found_files'] = raw_files
    with open(os.path.join(args.pred_dir, e['instance_id'] + '.json'), 'w') as f:
        json.dump(config, f, indent=2)


swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

with mp.Pool(16) as pool, tqdm(total=len(swe_bench_data), desc='localize file') as pbar:
    for return_contents in pool.imap_unordered(worker, swe_bench_data):
        pbar.update()

