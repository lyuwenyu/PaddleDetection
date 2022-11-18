
fleetrun --gpus 0,1,2,3 tools/train.py -c ./configs/dino/dino_r50_1x_coco.yml --eval &>dino_r50_1x_coco.txt 2>&1 &