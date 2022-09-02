import paddle

from ppdet.core.workspace import load_config
from ppdet.core.workspace import create
import ppdet

cfg = load_config('./ppyoloe_vit_base_vitdet_fpn_reader_yoloe_60e_coco.yml')
# print(cfg)

model = create(cfg.architecture)
lr = create('LearningRate')(10)
optimizer = create('OptimizerBuilder')(lr, model)

# for n, p in model.named_parameters():
#     if len(p.shape) == 1:
#         print(n, )

for i in range(60):
    for j in range(10):
        curr_lr = optimizer.get_lr()
        lr.step()
        print(i, j, curr_lr)

# LearningRateCNN:
#   base_lr: 0.01
#   schedulers:
#     - !CosineDecay
#       max_epochs: *epoch
#       # min_lr_ratio: 0.05
#       # last_plateau_epochs: 10
#     - !LinearWarmup
#       start_factor: 0.
#       epochs: 1

# OptimizerBuilderCNN:
#   optimizer:
#     type: Momentum
#     momentum: 0.9
#     use_nesterov: True
#   regularizer:
#     factor: 0.0005
#     type: L2

lr_decay = ppdet.optimizer.optimizer.CosineDecay(100, True)
lr_warmup = ppdet.optimizer.optimizer.LinearWarmup(start_factor=0, epochs=1)
lr_scheduler = ppdet.optimizer.LearningRate(
    base_lr=0.01, schedulers=[lr_decay, lr_warmup])(1000)

cfg_optimizer = {'type': 'Momentum', 'momentum': 0.9, 'use_nesterov': True}
cfg_regularizer = {'factor': 0.0005, 'type': 'L2'}

optimizer = ppdet.optimizer.OptimizerBuilder(
    regularizer=cfg_regularizer, optimizer=cfg_optimizer)(lr_scheduler, model)
