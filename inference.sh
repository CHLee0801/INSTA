# Inference command for P3 or BigBench
deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 63000 run.py --config run_configs/eval/t0_3b.json --target_cluster natural_language_inference --checkpoint_path google/t5-xl-lm-adapt

# Inference command for NIV2 or BBH
CUDA_VISIBLE_DEVICES=0 python evaluate_generation.py