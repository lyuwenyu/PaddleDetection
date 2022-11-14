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

