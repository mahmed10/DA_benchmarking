import os
import numpy as np
import torch
from PIL import Image
import yaml

from torchvision import transforms, datasets

def colorize_mask(mask):
	new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
	new_mask.putpalette(palette)
	return new_mask


class Relis3D(torch.utils.data.Dataset):
	def __init__(self, path_list, train = 'train', data_mode ='rgb', joint_transform=None, sliding_crop=None,
		transform=None, target_transform=None, dump_images=False, cv_split=None, eval_mode=False, 
		eval_scales=None, eval_flip=False):

		self.mode = train
		self.data_mode = data_mode
		self.joint_transform = joint_transform
		self.sliding_crop = sliding_crop
		self.transform = transform
		self.target_transform = target_transform
		self.dump_images = dump_images
		self.eval_mode = eval_mode
		self.eval_flip = eval_flip
		self.eval_scales = None

		if eval_scales != None:
			self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

		with open(path_list, "r") as file:
			self.imgs = file.readlines()

		self.masks = [
			path.replace("/pylon_camera_node/", "/pylon_camera_node_label_id/")
			.replace("/Rellis_3D_pylon_camera_node/", "/Rellis_3D_pylon_camera_node_label_id/")
			.replace(".jpg", ".png")
			for path in self.imgs
		]


		with open("./dataset/Rellis_3D.yaml", 'r') as stream:
			relis3dyaml = yaml.safe_load(stream)
		self.learning_map = relis3dyaml['learning_map']

		self.mean_std = ([0.54218053, 0.64250553, 0.56620195], [0.54218052, 0.64250552, 0.56620194])

	def convert_label(self, label, inverse=False):
		temp = label.copy()
		for k, v in self.learning_map.items():
			label[temp == k] = v
		return label

	# Till now did not use this but will use this in future
	def _eval_get_item(self, img, mask, scales, flip_bool):
		return_imgs = []
		for flip in range(int(flip_bool)+1):
			imgs = []
			if flip :
				img = img.transpose(Image.FLIP_LEFT_RIGHT)
			for scale in scales:
				w,h = img.size
				target_w, target_h = int(w * scale), int(h * scale) 
				resize_img =img.resize((target_w, target_h))
				tensor_img = transforms.ToTensor()(resize_img)
				final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
				imgs.append(tensor_img)
			return_imgs.append(imgs)
		return return_imgs, mask


	def __getitem__(self, index):
		img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()

		img = Image.open(img_path).convert('RGB')
		# img = img.resize((int(img.size[0]*0.75),int(img.size[1]*0.75)))
		mask = Image.open(mask_path)
		# mask = np.array(mask.resize((int(mask.size[0]*0.75),int(mask.size[1]*0.75))))

		mask = np.array(mask)

		mask = self.convert_label(mask)
		mask = mask.astype(np.uint8)

		img = np.asarray(img)
		img = img.transpose((2, 0, 1))

		return torch.as_tensor(img.copy()).float().contiguous(),torch.as_tensor(mask.copy()).long().contiguous()

	def __len__(self):
		return len(self.imgs)
