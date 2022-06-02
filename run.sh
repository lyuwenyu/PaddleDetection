
# ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 
# bash train.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_START_PORT=17000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

unset TRAINER_IP_LIST
unset TRAINER_INSTANCES
unset PADDLE_TRAINERS
unset TRAINER_IP_PORT_LIST
unset TRAINERS
unset TRAINERS_NUM
unset TRAINER_INSTANCES_NUM
unset TRAINER_HOSTS


cfg="configs/lst/lst_yoloe_l_3x.yml"
# cfg="configs/faster_rcnn/faster_rcnn_r50_vd_fpn_2x_coco.yml"


fleetrun \
--ips="10.127.6.17,10.127.5.142,10.127.45.13,10.127.44.151" \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py \
-c ${cfg} \
--fleet  \
--eval &>logs.txt 2>&1 & 

# --amp \
# -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams


