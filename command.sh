
ip_list="10.214.40.13,10.214.40.12"
fleetrun \
--ips=${ip_list} \
--selected_gpu 0,1,2,3 \
tools/train.py -c configs/vitdet/ppyoloe_vit_base_cae_msdcn_60e_coco.yml \
--eval &>logs.txt 2>&1 &


# python tools/export_model.py -c configs/vitdet/ppyoloe_r50_msdcn_36e_coco.yml -o weights=./4.pdparams trt=True exclude_nms=True
# python deploy/python/infer.py --model_dir=output_inference/ppyoloe_r50_msdcn_36e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=trt_fp16 --device=gpu --run_benchmark=True
# python deploy/python/infer.py --model_dir=output_inference/ppyoloe_r50_msdcn_36e_coco --image_file=demo/car.jpg --run_mode=paddle --device=gpu --run_benchmark=True
# paddle2onnx --model_dir=./output_inference/ppyoloe_r50_msdcn_36e_coco/ --model_filename=model.pdmodel --params_filename=model.pdiparams --opset_version 16 --save_file test.onnx
