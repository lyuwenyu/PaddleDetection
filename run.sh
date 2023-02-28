
# paddle2onnx --model_dir=./output_inference/dino_base_sim_speed/ --model_filename=model.pdmodel --params_filename=model.pdiparams  --opset_version 16 --save_file dino.onnx
# ../software/TensorRT-8.5.1.7/bin/trtexec --onnx=./dino.onnx --workspace=4096  --shapes=image:1x3x640x640,scale_factor:1x2,im_shape:1x2 --avgRuns=100 --fp16

python tools/export_model.py -c configs/dino/dino_base_sim_speed.yml -o weights=neck_1024.pdparams trt=True

rm -f shape_range_info.pbtxt

python deploy/python/infer.py --run_mode=trt_fp16 --device=GPU --run_benchmark=True --threshold=0.5 --output_dir=python_infer_output --image_file=./demo/000000014439_640x640.jpg --model_dir=output_inference/dino_base_sim_speed/

python deploy/python/infer.py --run_mode=trt_fp16 --device=GPU --run_benchmark=True --threshold=0.5 --output_dir=python_infer_output --image_file=./demo/000000014439_640x640.jpg --model_dir=output_inference/dino_base_sim_speed/