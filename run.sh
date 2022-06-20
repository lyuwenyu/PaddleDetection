# FLAGS_allocator_strategy=naive_best_fit
# FLAGS_use_system_allocator=1

ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

# python tools/eval.py -c configs/faster_rcnn/vit_base_16_faster_rcnn.yml -o weights=xxxx.pdparams

# python -m paddle.distributed.launch --log_dir=./logs --selected_gpu 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/vit_base_16_faster_rcnn.yml  --eval &>logs.txt 2>&1 &

python -m paddle.distributed.launch --log_dir=./logs --selected_gpu 0,1,2,3,4,5,6,7 tools/train.py -c ./configs/vitdet/cascade_rcnn_vit_base_16_hrfpn_coco.yml --amp --eval