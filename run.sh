# ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

# python tools/export_model.py -c configs/dino/dino_r50_1x_coco.yml -o weights=./4.pdparams trt=True
# paddle2onnx --model_dir=./output_inference/dino_r50_1x_coco/ --model_filename=model.pdmodel --params_filename=model.pdiparams  --opset_version 16 --save_file dino.onnx
# onnxsim dino.onnx dino_new.onnx  --overwrite-input-shape im_shape:1,2 image:1,3,640,640 scale_factor:1,2
# ../software/TensorRT-8.5.1.7/bin/trtexec --onnx=./dino_new.onnx --workspace=4096 --avgRuns=100  --fp16


fleetrun --gpus 0,1,2,3 tools/train.py -c ./configs/dino/dino_r50_1x_coco.yml --eval &>dino_r50_1x_coco.txt 2>&1 &

fleetrun --gpus 0,1,2,3 tools/train.py -c ./configs/dino/dino_r50_yoloe_reader_1x_coco.yml --eval &>dino_r50_yoloe_reader_1x_coco.txt 2>&1 &



# fleetrun --log_dir=./logs/ --gpus 0,1,2,3 tools/train.py -c ./configs/dino/dino_r50_640_pan_4_0_6_3x_coco.yml --eval &>dino_r50_640_pan_4_0_6_3x_coco_q_300_vd_ema_sync_bn.txt 2>&1 &

