# onnx                               1.12.0
# onnx-graphsurgeon                  0.3.19
# onnx-simplifier                    0.3.5
# onnxoptimizer                      0.3.2
# onnxruntime                        1.13.1
# onnxsim                            0.4.8
# paddle2onnx                        1.0.2

python tools/export_model.py -c configs/dino/dino_r50_1x_coco.yml -o weights=./4.pdparams 
paddle2onnx --model_dir=./output_inference/dino_r50_1x_coco/ --model_filename=model.pdmodel --params_filename=model.pdiparams  --opset_version 16 --save_file dino.onnx
onnxsim dino.onnx dino_new.onnx  --overwrite-input-shape im_shape:1,2 image:1,3,640,640 scale_factor:1,2
# python -m onnxoptimizer dino_new.onnx dino_new.onnx
../software/TensorRT-8.5.1.7/bin/trtexec --onnx=./dino_new.onnx --workspace=4096 --avgRuns=100  --fp16


# python deploy/python/infer.py  --run_mode=trt_fp16 --device=GPU  --run_benchmark True --threshold=0.5 --output_dir=python_infer_output --image_file=./demo/car.jpg --model_dir=output_inference/dino_r50_1x_coco



# cmake .. \
#   -DWITH_MKL=ON \
#   -DWITH_MKLDNN=ON \
#   -DWITH_GPU=ON \
#   -DWITH_TENSORRT=ON \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DCUDA_ARCH_NAME=Auto

#   -DWITH_DISTRIBUTE=ON \
#   -DWITH_TESTING=ON \
