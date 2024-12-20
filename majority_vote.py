import os
import random
import json
from tqdm import tqdm,trange
import multiprocessing as mp
from datasets import load_dataset

random.seed(42)
eval_ids = {'astropy__astropy-12907', 'astropy__astropy-13453', 'astropy__astropy-13579', 'astropy__astropy-14309', 'astropy__astropy-14539', 'astropy__astropy-14995', 'astropy__astropy-7166', 'astropy__astropy-7336', 'astropy__astropy-7606', 'astropy__astropy-7671', 'astropy__astropy-8872', 'django__django-10973', 'django__django-11066', 'django__django-11095', 'django__django-11099', 'django__django-11119', 'django__django-11133', 'django__django-11163', 'django__django-11179', 'django__django-11211', 'django__django-11292', 'django__django-11451', 'django__django-11551', 'django__django-11555', 'django__django-11603', 'django__django-11749', 'django__django-11790', 'django__django-11815', 'django__django-11848', 'django__django-11880', 'django__django-11964', 'django__django-11999', 'django__django-12039', 'django__django-12050', 'django__django-12143', 'django__django-12155', 'django__django-12193', 'django__django-12209', 'django__django-12276', 'django__django-12325', 'django__django-12419', 'django__django-12708', 'django__django-12713', 'django__django-12741', 'django__django-12754', 'django__django-12774', 'django__django-13012', 'django__django-13089', 'django__django-13109', 'django__django-13279', 'django__django-13315', 'django__django-13343', 'django__django-13363', 'django__django-13410', 'django__django-13417', 'django__django-13512', 'django__django-13569', 'django__django-13590', 'django__django-13658', 'django__django-13670', 'django__django-13741', 'django__django-13786', 'django__django-13807', 'django__django-13809', 'django__django-13820', 'django__django-13821', 'django__django-13837', 'django__django-13933', 'django__django-13964', 'django__django-14007', 'django__django-14017', 'django__django-14053', 'django__django-14089', 'django__django-14140', 'django__django-14238', 'django__django-14349', 'django__django-14351', 'django__django-14373', 'django__django-14434', 'django__django-14493', 'django__django-14500', 'django__django-14539', 'django__django-14559', 'django__django-14608', 'django__django-14672', 'django__django-14752', 'django__django-14765', 'django__django-14771', 'django__django-14787', 'django__django-14855', 'django__django-14915', 'django__django-14999', 'django__django-15037', 'django__django-15103', 'django__django-15104', 'django__django-15161', 'django__django-15277', 'django__django-15278', 'django__django-15280', 'django__django-15315', 'django__django-15368', 'django__django-15375', 'django__django-15380', 'django__django-15467', 'django__django-15499', 'django__django-15554', 'django__django-15561', 'django__django-15569', 'django__django-15572', 'django__django-15731', 'django__django-15741', 'django__django-15814', 'django__django-15851', 'django__django-15930', 'django__django-15987', 'django__django-16082', 'django__django-16116', 'django__django-16136', 'django__django-16139', 'django__django-16145', 'django__django-16255', 'django__django-16315', 'django__django-16333', 'django__django-16429', 'django__django-16454', 'django__django-16485', 'django__django-16493', 'django__django-16527', 'django__django-16569', 'django__django-16595', 'django__django-16612', 'django__django-16642', 'django__django-16661', 'django__django-16662', 'django__django-16801', 'django__django-16819', 'django__django-16899', 'django__django-16901', 'django__django-17029', 'django__django-7530', 'django__django-9296', 'matplotlib__matplotlib-13989', 'matplotlib__matplotlib-20859', 'matplotlib__matplotlib-22719', 'matplotlib__matplotlib-22865', 'matplotlib__matplotlib-23314', 'matplotlib__matplotlib-23412', 'matplotlib__matplotlib-23476', 'matplotlib__matplotlib-24026', 'matplotlib__matplotlib-24149', 'matplotlib__matplotlib-24570', 'matplotlib__matplotlib-24627', 'matplotlib__matplotlib-24637', 'matplotlib__matplotlib-25122', 'matplotlib__matplotlib-25287', 'matplotlib__matplotlib-25311', 'matplotlib__matplotlib-25332', 'matplotlib__matplotlib-25775', 'matplotlib__matplotlib-26113', 'matplotlib__matplotlib-26291', 'pallets__flask-5014', 'psf__requests-1724', 'psf__requests-1766', 'psf__requests-1921', 'psf__requests-2317', 'pydata__xarray-3151', 'pydata__xarray-3677', 'pydata__xarray-4075', 'pydata__xarray-4356', 'pydata__xarray-4629', 'pydata__xarray-4966', 'pydata__xarray-6461', 'pydata__xarray-6599', 'pydata__xarray-6744', 'pydata__xarray-7233', 'pylint-dev__pylint-4970', 'pylint-dev__pylint-6386', 'pylint-dev__pylint-6903', 'pytest-dev__pytest-5809', 'pytest-dev__pytest-6202', 'pytest-dev__pytest-7205', 'pytest-dev__pytest-7432', 'pytest-dev__pytest-7490', 'pytest-dev__pytest-7571', 'pytest-dev__pytest-7982', 'pytest-dev__pytest-8399', 'scikit-learn__scikit-learn-10297', 'scikit-learn__scikit-learn-10844', 'scikit-learn__scikit-learn-10908', 'scikit-learn__scikit-learn-11310', 'scikit-learn__scikit-learn-11578', 'scikit-learn__scikit-learn-12585', 'scikit-learn__scikit-learn-13124', 'scikit-learn__scikit-learn-13135', 'scikit-learn__scikit-learn-13142', 'scikit-learn__scikit-learn-13328', 'scikit-learn__scikit-learn-13439', 'scikit-learn__scikit-learn-13496', 'scikit-learn__scikit-learn-13779', 'scikit-learn__scikit-learn-14053', 'scikit-learn__scikit-learn-14141', 'scikit-learn__scikit-learn-14496', 'scikit-learn__scikit-learn-14710', 'scikit-learn__scikit-learn-14894', 'scikit-learn__scikit-learn-15100', 'scikit-learn__scikit-learn-25232', 'scikit-learn__scikit-learn-25931', 'scikit-learn__scikit-learn-25973', 'scikit-learn__scikit-learn-26323', 'scikit-learn__scikit-learn-9288', 'sphinx-doc__sphinx-10673', 'sphinx-doc__sphinx-7889', 'sphinx-doc__sphinx-7910', 'sphinx-doc__sphinx-8035', 'sphinx-doc__sphinx-8120', 'sphinx-doc__sphinx-8269', 'sphinx-doc__sphinx-8475', 'sphinx-doc__sphinx-8595', 'sphinx-doc__sphinx-8721', 'sphinx-doc__sphinx-9281', 'sphinx-doc__sphinx-9320', 'sphinx-doc__sphinx-9698', 'sphinx-doc__sphinx-9711', 'sympy__sympy-11618', 'sympy__sympy-12096', 'sympy__sympy-12481', 'sympy__sympy-13031', 'sympy__sympy-13372', 'sympy__sympy-13480', 'sympy__sympy-13647', 'sympy__sympy-13877', 'sympy__sympy-14531', 'sympy__sympy-14711', 'sympy__sympy-14976', 'sympy__sympy-15349', 'sympy__sympy-15809', 'sympy__sympy-16766', 'sympy__sympy-16886', 'sympy__sympy-17139', 'sympy__sympy-17318', 'sympy__sympy-18189', 'sympy__sympy-18763', 'sympy__sympy-19346', 'sympy__sympy-19637', 'sympy__sympy-19954', 'sympy__sympy-20154', 'sympy__sympy-20801', 'sympy__sympy-21847', 'sympy__sympy-22456', 'sympy__sympy-22914', 'sympy__sympy-23262', 'sympy__sympy-23534', 'sympy__sympy-23824', 'sympy__sympy-23950', 'sympy__sympy-24066', 'sympy__sympy-24213', 'sympy__sympy-24443', 'sympy__sympy-24539', 'sympy__sympy-24661'}
swe_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
all_instance_ids = [e['instance_id'] for e in swe_data]

def get_cache():
    patch_cache = {}
    report_cache = {}
    resolved_ids = set()
    for instance_id in tqdm(all_instance_ids,desc='get all outputs'):
        for subdir in os.listdir('evaluation/logs/run_evaluation'):
            if not subdir in patch_cache:
                patch_cache[subdir] = {}
            if not subdir in report_cache:
                report_cache[subdir] = {}
            if not os.path.isfile(f"evaluation/logs/run_evaluation/{subdir}/gemini/{instance_id}/patch.diff") or \
                    not os.path.isfile(f"evaluation/logs/run_evaluation/{subdir}/gemini/{instance_id}/report.json"):
                continue
            with open(f"evaluation/logs/run_evaluation/{subdir}/gemini/{instance_id}/patch.diff") as f:
                patch_cache[subdir][instance_id] = f.read().strip()
            with open(f"evaluation/logs/run_evaluation/{subdir}/gemini/{instance_id}/report.json") as f:
                report_cache[subdir][instance_id] = json.load(f)
            if report_cache[subdir][instance_id][instance_id]['resolved']:
                resolved_ids.add(instance_id)
    return patch_cache,report_cache,list(resolved_ids)

patch_cache,report_cache,resolved_instances = get_cache()
all_instance_ids = resolved_instances

def vote(directories):
    correct_set = set()
    correct_set_top3 = set()
    correct_set_top5 = set()
    correct_set_top10 = set()
    correct_set_top20 = set()
    correct_set_top50 = set()
    for instance_id in all_instance_ids:
        patch_counts = {}
        filtered_combinations = [d for d in directories if d in patch_cache and d in report_cache and instance_id in patch_cache[d] and instance_id in report_cache[d]]
        for subdir in filtered_combinations:
            patch = patch_cache[subdir][instance_id]
            report = report_cache[subdir][instance_id]
            if patch in patch_counts:
                if 'django' in subdir or 'matplotlib' in subdir:
                    patch_counts[patch][0] += 0.9
                else:
                    patch_counts[patch][0] += 1
                patch_counts[patch][1] = patch_counts[patch][1] or report[instance_id]['resolved']
            else:
                if 'django' in subdir or 'matplotlib' in subdir:
                    patch_counts[patch] = [0.9, report[instance_id]['resolved']]
                else:
                    patch_counts[patch] = [1,report[instance_id]['resolved']]
        if len(patch_counts)==0:
            continue
        cur_count = sorted(patch_counts.items(),key=lambda x:x[1][0],reverse=True)
        edit_lines_1 = []
        edit_lines_2 = []
        if len(cur_count) > 1:
            for l in cur_count[0][0].split('\n'):
                if l.startswith('+') or l.startswith('-'):
                    edit_lines_1.append(l)
            for l in cur_count[1][0].split('\n'):
                if l.startswith('+') or l.startswith('-'):
                    edit_lines_2.append(l)
        if set(edit_lines_1) == set(edit_lines_2) and len(edit_lines_1) > 0:
            if len(edit_lines_1) > len(edit_lines_2) and cur_count[0][1][1]:
                correct_set.add(instance_id)
                pass_flag = True
            elif len(edit_lines_1) <= len(edit_lines_2) and len(cur_count) > 1 and cur_count[1][1][1]:
                correct_set.add(instance_id)
                pass_flag = True
        elif cur_count[0][1][1] and (len(cur_count)==1 or cur_count[0][1][0]>cur_count[1][1][0]):
            correct_set.add(instance_id)
        elif len(cur_count)>1 and cur_count[0][1][0]==cur_count[1][1][0]:
            count1 = cur_count[0][0].count('+')+cur_count[0][0].count('-')
            count2 = cur_count[1][0].count('+')+cur_count[1][0].count('-')
            if count1<count2 and cur_count[0][1][1]:
                correct_set.add(instance_id)
            elif count1>=count2 and cur_count[1][1][1]:
                correct_set.add(instance_id)
        for idx in range(len(cur_count)):
            if cur_count[idx][1][1]:
                if idx<3:
                    correct_set_top3.add(instance_id)
                    correct_set_top5.add(instance_id)
                    correct_set_top10.add(instance_id)
                    correct_set_top20.add(instance_id)
                    correct_set_top50.add(instance_id)
                elif idx<5:
                    correct_set_top5.add(instance_id)
                    correct_set_top10.add(instance_id)
                    correct_set_top20.add(instance_id)
                    correct_set_top50.add(instance_id)
                elif idx<10:
                    correct_set_top10.add(instance_id)
                    correct_set_top20.add(instance_id)
                    correct_set_top50.add(instance_id)
                elif idx<20:
                    correct_set_top20.add(instance_id)
                    correct_set_top50.add(instance_id)
                elif idx<50:
                    correct_set_top50.add(instance_id)
    return {
        'Pass@1 in subset': len(correct_set.intersection(eval_ids)),
        'Pass@3 in subset': len(correct_set_top3.intersection(eval_ids)),
        'Pass@5 in subset': len(correct_set_top5.intersection(eval_ids)),
        'Pass@10 in subset': len(correct_set_top10.intersection(eval_ids)),
        'Pass@20 in subset': len(correct_set_top20.intersection(eval_ids)),
        'Pass@50 in subset': len(correct_set_top50.intersection(eval_ids)),
        'Pass@1': len(correct_set),
        'Pass@3': len(correct_set_top3),
        'Pass@5': len(correct_set_top5),
        'Pass@10': len(correct_set_top10),
        'Pass@20': len(correct_set_top20),
        'Pass@50': len(correct_set_top50),
    }

all_dirs = os.listdir('evaluation/logs/run_evaluation')
r = vote(all_dirs)
for k,v in r.items():
    if 'subset' in k:
        print(k, f"{v/259:.2%}")
    else:
        print(k, f"{v/500:.2%}")

