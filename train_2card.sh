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
                                                                                                                    

cfg="configs/lst/lst_yoloe_s_3x.yml"
# cfg="configs/faster_rcnn/faster_rcnn_r50_vd_fpn_2x_coco.yml"
fleetrun="/root/paddlejob/workspace/env_run/lvwenyu01/anaconda3/bin/fleetrun"


${fleetrun} \
--ips="10.127.6.17,10.127.5.142" \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py -c ${cfg} \
--eval &>logs.txt 2>&1 &


# -r output/lst_yoloe_s_3x/53.pdparams \
