import os
import numpy as np
import torch
from PIL import Image
import yaml
import data.edge_utils as edge_utils

num_classes = 19


class Relis3D(torch.utils.data.Dataset):
	def __init__(self, train, path_list, data_mode):
		self.train = train
		self.data_mode = data_mode

		with open(path_list, "r") as file:
			self.imgs = file.readlines()
		#print("\nImage path files:")
		#print(self.imgs)

		self.masks = [
			path.replace("/pylon_camera_node/", "/pylon_camera_node_label_id/")
			.replace("/Rellis_3D_pylon_camera_node/", "/Rellis_3D_pylon_camera_node_label_id/")
			.replace(".jpg", ".png")
			for path in self.imgs
		]
		#print("\nImage mask path files:")
		#print(self.masks)


		with open("./dataset/Rellis_3D.yaml", 'r') as stream:
			relis3dyaml = yaml.safe_load(stream)
		self.learning_map = relis3dyaml['learning_map']

	def convert_label(self, label, inverse=False):
		temp = label.copy()
		for k, v in self.learning_map.items():
			label[temp == k] = v
		return label


	def __getitem__(self, index):
		img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip() 
		#print("img_path path: ", img_path)
		#print("mask_path path: ", mask_path)
		img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

		mask = np.array(mask)
		mask_copy = self.convert_label(mask)
		mask = Image.fromarray(mask_copy.astype(np.uint8))

		_edgemap = mask_copy
		del mask_copy
		_edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)

		_edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)

		edgemap = torch.from_numpy(_edgemap).float()
		return img, mask, edgemap

	def __len__(self):
		return len(self.imgs)
