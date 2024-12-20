## Setup
Follow the [guidance in SWE-bench](https://github.com/princeton-nlp/SWE-bench?tab=readme-ov-file#-set-up) to set up docker and conda environments.
Install the following packages:
```bash
pip install libcst
pip install vertexai
```

Download all [repository files](https://drive.google.com/file/d/1cUUwps4kEV52dCmtQxGIIFM4rCROk6mn/view?usp=sharing&resourcekey=0-0aa7hLvVOjCzv2DatBgcAA) and [repository structures](https://drive.google.com/file/d/1VBnKwab3fl3EIue4yu2LBILXAiD9aqov/view?usp=sharing&resourcekey=0-CFu-RBq2ov-ekWxxmBF34A), and put them under the inference folder.

The Gemini calling function `request_gemini_engine` is in `inference/utils.py`. We currently use the API Vertexai. Change it if it is different.

## Inference
The inference is done under the `inference` folder.
```bash
cd inference
```

#### Step 1: Identify suspicious files to modify
```bash
python localize_file.py --pred_dir {result_folder}
```

#### Step 2: Identify suspicious classes, functions or variables from retrieved files in step 1
```bash
python localize_function.py --pred_dir {result_folder}
```

#### Step 3: Make edits based on the retrieved code pieces in step 2
```bash
python edit.py --pred_dir {result_folder} --gemini_version {trained_checkpoint_id}
```

#### Step 4: Parse the generated outputs into diff formats
```bash
python post_process.py --pred_dir {result_folder} --output_path {output_path}
```

The trained checkpoint id can be chosen from the following:
* `projects/618488765595/locations/us-central1/tuningJobs/5189087506007588864`: Trained with 3k synthesized data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/5248990823634173952`: Trained with 3k synthesized data for 2 epochs.
* `projects/618488765595/locations/us-central1/tuningJobs/4976151686125977600`: Trained with 9k synthesized data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/4724000409650200576`: Trained with 20k synthesized data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/7671606365764190208`: Trained with 24k synthesized data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/6343387523317760000`: Trained with 28k synthesized data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/6869814998999236608`: Trained with 1k training data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/889879118781349888`: Trained with 2k training data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/7311378868714078208`: Trained with 4k training data for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/2758872964140105728`: Trained with 4k training data (different sampling) for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/7768621774241005568`: Trained with 12k synthesized data on Django for 1 epoch.
* `projects/618488765595/locations/us-central1/tuningJobs/6564682105471631360`: Trained with 2k synthesized data on Matplotlib for 1 epoch.

## Evaluation
```bash
cd evaluation
python -m swebench.harness.run_evaluation --predictions_path {output_path} --max_workers 24 --run_id {experiment_id}
```
Known issue: in many cases, the evaluation seems to never end. We check the output files and terminate the process if no new instance is evaluated after a long period of time, e.g., 20 minutes.
```bash
python view_results.py --run_id {experiment_id}
```

## Majority vote
```bash
python majority_vote.py
```
It looks for sub-directories under `evaluation/logs/run_evaluation` and perform majority vote based on the generated patches. 
In current experiments, we sample each checkpoint 10 times (full pass from step 1 to step 4), which results in 120 generations per example.

## End-to-end evaluation
To run all inference and evaluations end-to-end, you may use run.sh
```bash
tmux new -s mysession
./run.sh
```





