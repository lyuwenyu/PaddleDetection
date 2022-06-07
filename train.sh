ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

fleetrun --gpus 0,1,2,3  tools/train.py -c configs/picodet/picodetv2_s_416_coco.yml --eval