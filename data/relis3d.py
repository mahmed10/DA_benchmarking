import os
import numpy as np
import torch
from PIL import Image
import yaml
import data.edge_utils as edge_utils

num_classes = 2
ignore_label = 0

def colorize_mask(mask):
	# mask: numpy array of the mask
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

		self.mean_std = ([0.54218053, 0.64250553, 0.56620195], [0.54218052, 0.64250552, 0.56620194])

	def convert_label(self, label, inverse=False):
		temp = label.copy()
		for k, v in self.learning_map.items():
			label[temp == k] = v
		return label

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
		#print("img_path path: ", img_path)
		#print("mask_path path: ", mask_path)
		img = Image.open(img_path).convert('RGB')
		img = img.resize((int(img.size[0]*0.75),int(img.size[1]*0.75)))
		mask = Image.open(mask_path)
		mask = np.array(mask.resize((int(mask.size[0]*0.75),int(mask.size[1]*0.75))))


		mask = mask[:, :]
		print(mask)

		mask_copy = self.convert_label(mask)

		mask = Image.fromarray(mask_copy.astype(np.uint8))
		# Image Transformations


		# if self.joint_transform is not None:
		# 	img, mask = self.joint_transform(img, mask)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			mask = self.target_transform(mask)
		if self.mode == 'test':
			return img, mask, img_name, item['img']

		_edgemap = mask.numpy()
		_edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)

		_edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)

		edgemap = torch.from_numpy(_edgemap).float()
		# img = torch.from_numpy(img).float()
		# mask = torch.from_numpy(mask).float()

		# Debug
		if self.dump_images:
			outdir = '../../dump_imgs_{}'.format(self.mode)
			os.makedirs(outdir, exist_ok=True)
			out_img_fn = os.path.join(outdir, img_name + '.png')
			out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
			mask_img = colorize_mask(np.array(mask))
			img.save(out_img_fn)
			mask_img.save(out_msk_fn)
		print(mask)
		return img, mask, edgemap

	def __len__(self):
		return len(self.imgs)
