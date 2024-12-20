import os
import json
import shutil
import argparse
import subprocess
from datasets import load_dataset

swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
swe_bench_data = [e for e in swe_bench_data]
parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, default='results')
parser.add_argument("--output_path", type=str, default='../evaluation/output.jsonl')
args = parser.parse_args()
if not os.path.isdir(os.path.join(args.pred_dir,'repairs_processed')):
    os.makedirs(os.path.join(args.pred_dir,'repairs_processed'))

def parse_output(s):
    components = s.split('```python')
    replacements = []
    for c in components:
        replacements += c.split('```')[0].split('>>>>>>> REPLACE')
    replacements = [r.strip() for r in replacements if r.strip()!='']
    return replacements

def worker(e):
    if not os.path.isfile(os.path.join(args.pred_dir, 'repairs', e['instance_id'] + '.json')):
        return False
    with open(os.path.join(args.pred_dir, 'repairs', e['instance_id'] + '.json')) as f:
        outputs = json.load(f)
    for replacements in outputs['raw_output']:
        if os.path.isdir(os.path.join('all_repos', e['instance_id'])):
            shutil.rmtree(os.path.join('all_repos', e['instance_id']))
        replacements = parse_output(replacements)
        processed_replace = []
        for r in replacements:
            try:
                lines = r.split('\n')
                file_name = lines[0].replace('#','').strip()
                components = '\n'.join(lines[2:]).split('=======')
                assert len(components)==2
                original_code = components[0]
                modified_code = components[1]
                if not os.path.isfile(os.path.join('all_repos_verified',e['instance_id'],file_name)) or original_code.strip()=='':
                    continue
                processed_replace.append({
                    'file': file_name,
                    'original_code': original_code,
                    'modified_code': modified_code
                })
            except Exception as error:
                pass
        repo_dir = os.path.join('all_repos',e['instance_id'])
        os.makedirs(repo_dir)
        subprocess.run(f"cd {repo_dir} && git init", shell=True)
        for r in processed_replace:
            cur_file_path = os.path.join(repo_dir,r['file'])
            if not os.path.isdir(os.path.dirname(cur_file_path)):
                subprocess.run(f"mkdir -p {os.path.dirname(cur_file_path)}", shell=True)
            if not os.path.isfile(cur_file_path):
                shutil.copyfile(os.path.join('all_repos_verified',e['instance_id'],r['file']),cur_file_path)
                subprocess.run(f"cd {repo_dir} && git add {r['file']} && git commit -m 'initial commit'",shell=True,)
            with open(cur_file_path) as f:
                old_content = f.read()
            new_content = old_content.replace(r['original_code'].strip(),r['modified_code'].strip())
            with open(cur_file_path,'w') as f:
                f.write(new_content)

        o = subprocess.run(f"cd all_repos/{e['instance_id']} && git diff", shell=True, capture_output=True)
        s = o.stdout.decode("utf-8")
        with open(os.path.join(args.pred_dir,'repairs_processed',e['instance_id']+'.json'),'w') as f:
            json.dump({
                'instance_id': e['instance_id'],
                'model_patch': s,
                'model_name_or_path': 'gemini'
            },f,indent=2)
        if s.strip()!='':
            return True
    return False

modified = 0
for e in swe_bench_data:
    return_content = worker(e)
    if return_content:
        modified += 1

final_outputs = []
for e in swe_bench_data:
    if os.path.isfile(os.path.join(args.pred_dir,'repairs_processed',e['instance_id']+'.json')):
        with open(os.path.join(args.pred_dir,'repairs_processed',e['instance_id']+'.json')) as f:
            final_outputs.append(json.load(f))
output_dir = os.path.dirname(args.output_path)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
with open(args.output_path,'w') as f:
    for o in final_outputs:
        f.write(json.dumps(o).strip()+'\n')


