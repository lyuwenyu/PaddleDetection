

 python tools/export_model.py -c configs/dino/dino_base_sim_speed.yml -o weights=neck_1024.pdparams trt=True

python deploy/python/infer.py --run_mode=trt_fp16 --device=GPU --run_benchmark=True --threshold=0.5 --output_dir=python_infer_output --image_file=./demo/000000014439_640x640.jpg --model_dir=output_inference/dino_base_sim_speed/