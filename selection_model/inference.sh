CUDA_VISIBLE_DEVICES=5 python inference_max_of_max.py more_filter_1130/173 > more_1130_epoch0.txt &
sleep 2
CUDA_VISIBLE_DEVICES=1 python inference_max_of_max.py more_filter_1130/346 > more_1130_epoch11.txt &
sleep 2
CUDA_VISIBLE_DEVICES=2 python inference_max_of_max.py more_filter_1130/519 > more_1130_epoch2.txt &
sleep 2
CUDA_VISIBLE_DEVICES=3 python inference_max_of_max.py more_filter_1130/692 > more_1130_epoch3.txt &
sleep 2
CUDA_VISIBLE_DEVICES=4 python inference_max_of_max.py more_filter_1130/865 > more_1130_epoch4.txt 