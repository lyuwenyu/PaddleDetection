
# ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolox/yolox_l_120e_coco.yml  --eval &> yolox_l_120e.txt 2>&1 &

