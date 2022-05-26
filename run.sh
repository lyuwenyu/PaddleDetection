git clone https://github.com/lyuwenyu/PaddleDetection.git
cd PaddleDetection
git checkout large_scale_model_v1_L
pip install -r requirements.txt 


fleetrun \
--ips="ip1,ip2" \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py \
--fleet  \
-c configs/lst/lst_yoloe_l_3x.yml \
-r https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams \
--eval &>logs.txt 2>&1 & 


mkdir -p ~/dataset 
cd ~/dataset 
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/cache.pkl


mkdir -p ~/dataset/coco/
cd ~/dataset/coco/
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/coco/train2017.zip
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/coco/val2017.zip
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/coco/annotations.zip
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/coco/annotations/instances_train2017.json.csv
ls *.zip | xargs -n1 unzip -q 
mv instances_train2017.json.csv annotations/



mkdir -p ~/dataset/obj365/
cd ~/dataset/obj365/
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/obj365/train/train.tar
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/obj365/val/val.tar
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/obj365/annos/annos.tar
# wget --no-proxy http://10.21.226.186:8787/workspace/dataset/obj365/annos/zhiyuan_objv2_val.json
# wget --no-proxy http://10.21.226.186:8787/workspace/dataset/obj365/annos/zhiyuan_objv2_train.json
# wget --no-proxy http://10.21.226.186:8787/workspace/dataset/obj365/annos/zhiyuan_objv2_train.json.csv
mkdir annos; tar -xvf annos.tar -C ./annos

mkdir train; tar -xvf train.tar -C ./train 
cd train; ls *.tar.gz | xargs -n1 tar -xf; cd ..


# mkdir val
# tar -xvf val.tar -C ./val  

# mkdir annos 
# mv *.json ./annos
# mv *.csv ./annos



mkdir -p ~/dataset/oid/full
cd ~/dataset/oid/full
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/annos.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/test.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_0.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_1.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_2.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_3.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_4.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_5.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_6.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_7.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_8.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_9.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_a.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_b.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_c.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_d.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_e.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/train_f.tar.gz
wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/full/validation.tar.gz
# wget --no-proxy http://10.21.226.186:8787/workspace/dataset/oid/annos/oidv6-train-annotations-bbox.csv-tmp.csv
# ls *.tar.gz | xargs -n1 tar -xf

# mkdir train
# mv train_*/* ./train
# find train_*/ -name "*.jpg" | xargs -i cp {} train
# find train_*/ -name "*.jpg" | xargs -i mv {} train/

mkdir train
ls train_*.tar.gz | xargs -i tar -xvf {} --strip-components 1 -C train/



