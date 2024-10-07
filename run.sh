nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run.py --config run_configs/p3/insta_aligned/anli_top_5.json > out/anli_top_5.txt &

