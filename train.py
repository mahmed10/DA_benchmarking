import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
import numpy as np

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm

import data

from models.unet import UNET
from models.deeplabv2 import get_deeplab_v2

from evaluate import iou_calculation
from domain_adaptation.train_UDA import train_domain_adaptation
from domain_adaptation.train_Sonly import train_source_only

from torchvision import transforms

parser = argparse.ArgumentParser(description='Benchmarking')
# parser.add_argument('--da_mode', type=str, default='SingleDA_')
parser.add_argument('--da_mode', type=str, default='Sourceonly_')
parser.add_argument('--branch_mode', type=str, default='SingleBranch_')
parser.add_argument('--class_mode', type=str, default='CommonClasses_')
parser.add_argument('--source_dataset', type=str, default='rellis3d')
parser.add_argument('--target_dataset', type=str, default='semantickitti')
parser.add_argument('--train_mode', type=str, default='train')
parser.add_argument('--train_path_list', type=str, default='./dataset/Rellis3d/trainlist.txt')
parser.add_argument('--val_path_list', type=str, default='./dataset/SemanticKitti/vallist.txt')
parser.add_argument('--data_mode', type=str, default='rgb')
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--image_width', type=int, default=572)
parser.add_argument('--image_height', type=int, default=572)
parser.add_argument('--num_classes', type=int, default=9)
parser.add_argument('--train_data_size', type=int, default=1000)
parser.add_argument('--model_path', type=str, default='./checkpoints/')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--multi_level', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--start_epoches', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0005)

args = parser.parse_args()

model_path = args.model_path+args.da_mode+args.branch_mode+args.class_mode+'_Source_'+args.source_dataset+'_Target_'+args.target_dataset

try:
	os.mkdir(model_path)
except OSError:
	print(os.path.abspath(model_path) + ' folder already existed')
else:
	print(os.path.abspath(model_path) + ' folder created')
args.model_path = model_path+'/'
del model_path

epoches = args.epoches

if torch.cuda.is_available():
	args.device = 'cuda:0'
	torch.cuda.empty_cache()
else:
	args.device = "cpu"

def check_dataloader(loader, source):
	targetloader_iter = enumerate(loader)
	for i in range(7):
		_, batch = targetloader_iter.__next__()
		img, mask= batch
		trans = transforms.ToPILImage()
		plt.figure()
		plt.title('Image')
		plt.imshow(trans(img[0]))
		plt.savefig(args.model_path+source+'_image'+repr(i)+'.png')
		plt.figure()
		plt.title('Mask GT')
		values = np.unique(mask[0].ravel())
		im = plt.imshow(mask[0])
		colors = [ im.cmap(im.norm(value)) for value in values]
		patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
		plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
		plt.savefig(args.model_path+source+'_mask'+repr(i)+'.png')


def main():
	args.source_loader = data.setup_loaders(args.source_dataset, args.train_path_list, args.batch_size)
	check_dataloader(args.source_loader, 'source')
	args.target_loader = data.setup_loaders(args.target_dataset, args.val_path_list, args.batch_size)
	check_dataloader(args.target_loader, 'target')
	print('Data Loaded Successfully!')

	args.model = UNET(in_channels=args.in_channels, classes=args.num_classes).to(args.device)
	if(args.da_mode == 'SingleDA_'):
		train_domain_adaptation(args)
	if(args.da_mode == 'Sourceonly_'):
		train_source_only(args)

if __name__ == '__main__':
	main()