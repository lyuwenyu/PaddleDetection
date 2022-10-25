# python tools/export_model.py -c configs/vitdet/ppyole_vit_base_coco.yml
# python deploy/python/infer.py  --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=python_infer_output --image_dir=./demo --run_benchmark True --model_dir=output_inference/

# python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/vitdet/ppyole_vit_base_60e_coco.yml --eval &> train.txt 2>&1 &

# ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 




python deploy/python/infer.py  --run_mode=paddle --device=GPU  --run_benchmark True --threshold=0.5 --output_dir=python_infer_output --image_dir=./demo --model_dir=output_inference/ 