# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/anli_top_5.json > out/anli_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/cb_top_5.json > out/cb_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/copa_top_5.json > out/copa_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/hella_top_5.json > out/hella_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/rte_top_5.json > out/rte_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/story_top_5.json > out/story_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/wic_top_5.json > out/wic_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/wino_top_5.json > out/wino_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p2/wsc_top_5.json > out/wsc_top_5.txt &



# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/anli_top_5.json > out/anli_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/cb_top_5.json > out/cb_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/copa_top_5.json > out/copa_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/hella_top_5.json > out/hella_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/rte_top_5.json > out/rte_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/story_top_5.json > out/story_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/wic_top_5.json > out/wic_top_5.txt &&
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 63000 run_p3.py --config run_configs/p3/aibts_p4/wino_top_5.json > out/wino_top_5.txt &
# nohup deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_p3.py --config run_configs/p3/aibts_p4/wsc_top_5.json > out/wsc_top_5.txt &


# deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_polyglot.py --config run_configs/polyglot.json

# deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 63000 run_exaone.py --config run_configs/exaone.json

# deepspeed --include localhost:8,9,10,11 --master_port 63000 run_exaone.py --config run_configs/exaone.json
# sleep 6h
deepspeed --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --master_port 64000 run_polyglot.py --config run_configs/polyglot.json

deepspeed --include localhost:0 --master_port 64000 run_p3.py --config run_configs/debug.json