import os
import glob
import random

path = './dataset/SemanticKitti/kitti-step/panoptic_maps/train/*/*.png'

#dirs = os.listdir(path1)
dirs=glob.glob(path)
random.shuffle(dirs)
for i in range(6000):
    print(dirs[i].replace("/kitti-step/panoptic_maps/","/data_tracking_image_2/").replace("/train/","/training/image_02/"))