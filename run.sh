#cd inference
#
#python localize_file.py --pred_dir results
#
#python localize_function.py --pred_dir results
#
#python edit.py --pred_dir results --gemini_version projects/618488765595/locations/us-central1/tuningJobs/5189087506007588864
#
#python post_process.py --pred_dir results --output_path ../evaluation/outputs/test.jsonl
#
#cd ../evaluation
#
#python -m swebench.harness.run_evaluation --predictions_path outputs/test.jsonl --max_workers 24 --run_id test1


for checkpoint in 5189087506007588864 5248990823634173952 4976151686125977600 4724000409650200576 7671606365764190208 6343387523317760000 6869814998999236608 889879118781349888 7311378868714078208 2758872964140105728 7768621774241005568 6564682105471631360
do
    for i in $(seq 1 10)
    do
        echo $checkpoint
        echo $i
        tmux new-window -t mysession: -n "run0" "source activate agentless && cd inference && python localize_file.py --pred_dir results_${checkpoint}_${i} && python localize_function.py --pred_dir results_${checkpoint}_${i} && python edit.py --pred_dir results_${checkpoint}_${i} --gemini_version projects/618488765595/locations/us-central1/tuningJobs/${checkpoint} && python post_process.py --pred_dir results_${checkpoint}_${i} --output_path ../evaluation/outputs/${checkpoint}_${i}.jsonl && cd ../evaluation && python -m swebench.harness.run_evaluation --predictions_path outputs/${checkpoint}_${i}.jsonl --max_workers 24 --run_id ${checkpoint}_${i}; exec bash"
        sleep 3600
        tmux kill-window -t run0
    done
done