import os
import glob
import random

path = './dataset/CityScapes/leftImg8bit/train/*/*.png'

#dirs = os.listdir(path1)
dirs=glob.glob(path)
random.shuffle(dirs)
for i in range(6000):
    print(dirs[i])