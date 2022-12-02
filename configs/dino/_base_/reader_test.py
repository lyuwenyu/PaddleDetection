import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg = load_config('./configs/dino/_base_/reader_test.yml')
reader = create('TrainReader')

print(reader)
