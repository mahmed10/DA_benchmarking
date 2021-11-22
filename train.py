import os
import argparse

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm

import data

from models.unet import UNET

from evaluate import iou_calculation


parser = argparse.ArgumentParser(description='Benchmarking')
parser.add_argument('--dataset', type=str, default='rellis3d')
parser.add_argument('--train_mode', type=str, default='train')
parser.add_argument('--train_path_list', type=str, default='trainlist.txt')
parser.add_argument('--val_path_list', type=str, default='vallist.txt')
parser.add_argument('--data_mode', type=str, default='rgb')
parser.add_argument('--model_path', type=str, default='./checkpoints/')
parser.add_argument('--load_model', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.0005)

args = parser.parse_args()

model_path = args.model_path+'SourceOnlyTrain_'+args.dataset

try:
	os.mkdir(model_path)
except OSError:
	print(os.path.abspath(model_path) + ' folder already existed')
else:
	print(os.path.abspath(model_path) + ' folder created')
model_path = args.model_path+'SourceOnlyTrain_'+args.dataset+'/'

epoches = args.epoches

if torch.cuda.is_available():
	args.device = 'cuda:0'
	torch.cuda.empty_cache()
else:
	args.device = "cpu"

def train(data, model, optimizer, loss_fn, device):
	print('Entering into train function')
	loss_values = []
	data = tqdm(data)
	for index, batch in enumerate(data): 
		X, y = batch
		# print(X.size())
		# print(y.size())
		# print(X)
		# print(y)
		X, y = X.to(device), y.to(device)
		preds = model(X)
		loss = loss_fn(preds, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return loss.item()

def main():
	global epoch
	epoch = 0 
	train_set = data.setup_loaders(args.dataset, args.train_path_list, args.batch_size)
	val_set = data.setup_loaders(args.dataset, args.val_path_list, args.batch_size)
	print('Data Loaded Successfully!')
	loss_vals = []
	train_ious = []
	val_ious = []

	# Defining the model, optimizer and loss function
	unet = UNET(in_channels=3, classes=10).to(args.device).train()
	optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)
	loss_function = nn.CrossEntropyLoss(ignore_index=255)

	if args.load_model == True:
		checkpoint = torch.load(model_path)
		unet.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optim_state_dict'])
		epoch = checkpoint['epoch']+1
		loss_vals = checkpoint['loss_values']
		train_ious = checkpoint['train_iou']
		print("Model successfully loaded!") 

	for e in range(epoch, epoches):
		print(f'Epoch: {e}')
		loss_val = train(train_set, unet, optimizer, loss_function, args.device)
		loss_vals.append(loss_val)
		iou =  iou_calculation(train_set, unet,args.device)
		train_ious.append(iou)
		print('train iou', iou)
		# iou =  iou_calculation(val_set, unet,args.device)
		# val_ious.append(iou)
		# print('val iou', iou)
		torch.save({
			'model_state_dict': unet.state_dict(),
			'optim_state_dict': optimizer.state_dict(),
			'epoch': e,
			'loss_values': loss_vals,
			'train_iou': train_ious
		}, model_path + 'epoch_'+str(e).zfill(4))
		# torch.save({
		# 	'val_iou' : val_ious
		# }, model_path + 'val_iou_'+args.dataset)
		print("Epoch completed and model successfully saved!")

if __name__ == '__main__':
	main()