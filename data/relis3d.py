import os
import numpy as np
import torch
from PIL import Image

class Relis3D(torch.utils.data.Dataset):
	def __init__(self, train, path_list):
		self.train = train

		with open(path_list, "r") as file:
			self.imgs = file.readlines()

		self.masks = [
			path.replace("pylon_camera_node", "pylon_camera_node_label_id").replace(".jpg", ".png")
			for path in self.imgs
		]

	def __getitem__(self, index):
		img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()
		img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

		return img, mask

	def __len__(self):
		return len(self.imgs)
