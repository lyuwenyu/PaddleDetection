import paddle

from ppdet.core.workspace import load_config
from ppdet.core.workspace import create

cfg = load_config('./ppyole_vit_base_reader_yoloe_60e_coco.yml')
# print(cfg)

model = create(cfg.architecture)
lr = create('LearningRate')(10)
optimizer = create('OptimizerBuilder')(lr, model)

for i in range(60):
    for j in range(10):
        curr_lr = optimizer.get_lr()
        lr.step()

        print(i, j, curr_lr)
