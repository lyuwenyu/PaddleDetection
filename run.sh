
ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 


fleetrun \
--ips="ip1,ip2,ip3" \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py \
-c configs/lst/lst_yoloe_l_3x.yml \
# -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
--fleet  \
--amp \
--eval &>logs.txt 2>&1 & 


