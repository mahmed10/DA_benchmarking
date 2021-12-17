import os
import glob
import random

path = './dataset/Rellis3d/Rellis_3D_pylon_camera_node_label_id/Rellis-3D/*/pylon_camera_node_label_id/*.png'

#dirs = os.listdir(path1)
dirs=glob.glob(path)
random.shuffle(dirs)
for i in range(200):
    print(dirs[i].replace("/pylon_camera_node_label_id/","/pylon_camera_node/").replace("/Rellis_3D_pylon_camera_node_label_id/","/Rellis_3D_pylon_camera_node/").replace(".png",".jpg"))