# ! /usr/bin/env python
# -*- coding:utf-8 -*-

# @author: Edison Jia-hao-Chen
# time: 2019-5-31
# email: JiahaoChen@whu.edu.cn

from mmdet.apis import init_detector, inference_detector, show_result

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
checkpoint_file = '/home/edison/Downloads/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
# 初始化模型
model = init_detector(config_file, checkpoint_file)

# 测试一张图片
img = '/home/edison/tools/detectron/detectron/demo/16004479832_a748d55f21_k.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES, out_file='/home/edison/workspace/zmvision-outdir/test_mmdet_demo2_531.jpg')

# 测试一系列图片
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))