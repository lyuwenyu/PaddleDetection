
# ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                                                                         
export FLAGS_START_PORT=17000      
                                                                             
                                                                                     
unset TRAINER_IP_LIST                                                                                               
unset TRAINER_INSTANCES                                                                                             
unset PADDLE_TRAINERS                                                                                               
unset TRAINER_IP_PORT_LIST                                                                                          
unset TRAINERS                                                                                                      
unset TRAINERS_NUM                                                                                                  
unset TRAINER_INSTANCES_NUM                                                                                         
unset TRAINER_HOSTS                                                                                                 

fleetrun \
--ips="10.127.6.17,10.127.5.142,10.127.45.13,10.127.44.151" \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py \
-c configs/lst/lst_yoloe_l_3x.yml \
# -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
--fleet  \
--amp \
--eval &>logs.txt 2>&1 & 


