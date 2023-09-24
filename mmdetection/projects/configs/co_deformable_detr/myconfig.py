_base_ = "./co_deformable_detr_r50_1x_coco.py"

dataset_type = 'CocoDataset'
classes = ("needle driver", "force bipolar", "cadiere forceps", "monopolar curved scissor", "bipolar forceps",
           "grasping retractor", "vessel sealer", "clip applier", "prograsp forceps", "tip up fenestrated grasper", 
           "permanent cautery hook spatula", "suction irrigator", "stapler", "bipolar dissector")

import mmdet
mmdet.datasets.coco.CocoDataset.CLASSES=classes
# model = dict(bbox_head=dict(num_classes=14))

data = dict(
    train=dict(type=dataset_type, classes=classes),
    val=dict(type=dataset_type, classes=classes),
    test=dict(type=dataset_type, classes=classes)
    )

data_root = "data/"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/train.json',
        img_prefix=data_root + 'imgs/',),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/val.json',
        img_prefix=data_root + 'imgs/',),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/val.json',
        img_prefix=data_root + 'imgs/',))


load_from = "work_dirs/latest.pth"

runner = dict(type='EpochBasedRunner', max_epochs=10)