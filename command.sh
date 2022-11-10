
ip_list="10.214.40.13,10.214.40.12"
fleetrun \
--ips=${ip_list} \
--selected_gpu 0,1,2,3 \
tools/train.py -c configs/vitdet/ppyoloe_vit_base_cae_msdcn_60e_coco.yml \
--eval &>logs.txt 2>&1 &
