import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg = load_config(
    './configs/dino/dino_r50_640_pan_4_0_6_3x_coco_reader_enhance_distill.yml')

# model = create('DETR')
# print(model)
# print(cfg.teacher['architecture'])

teacher = create(cfg.teacher['architecture'])
head = create(cfg.teacher['head'])

print(cfg.teacher)

# if 'pretrain_weights' in cfg.teacher:
#     state = paddle.load(cfg.teacher['pretrain_weights'])
#     teacher.set_state_dict(state)
#     teacher.eval()
