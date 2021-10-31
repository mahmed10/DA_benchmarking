import os
import argparse
import numpy as np
from PIL import Image

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms, datasets, utils
from tqdm import tqdm

import data

from model import UNET


parser = argparse.ArgumentParser(description='Benchmarking')
parser.add_argument('--dataset', type=str, default='relis3d')
parser.add_argument('--train_mode', type=str, default='train')
parser.add_argument('--train_path_list', type=str, default='trainlist.txt')
parser.add_argument('--val_path_list', type=str, default='vallist.txt')
parser.add_argument('--data_mode', type=str, default='rgb')
parser.add_argument('--model_path', type=str, default='YOUR-MODEL-PATH')
parser.add_argument('--load_model', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoches', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.0005)

args = parser.parse_args()

epoches = args.epoches

if torch.cuda.is_available():
	args.device = 'cuda:0'
else:
	args.device = "cpu"

def train(data, model, optimizer, loss_fn, device):
	print('Entering into train function')
	loss_values = []
	data = tqdm(data)
	for index, batch in enumerate(data): 
		X, y = batch
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
	train_set = data.setup_loaders(args.train_path_list, args.batch_size)
	print('Data Loaded Successfully!')
	loss_vals = []

	# Defining the model, optimizer and loss function
	unet = UNET(in_channels=3, classes=20).to(args.device).train()
	optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)
	loss_function = nn.CrossEntropyLoss(ignore_index=255)

	if args.load_model == True:
		checkpoint = torch.load(args.model_path)
		unet.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optim_state_dict'])
		epoch = checkpoint['epoch']+1
		loss_vals = checkpoint['loss_values']
		print("Model successfully loaded!") 

	for e in range(epoch, epoches):
		print(f'Epoch: {e}')
		loss_val = train(train_set, unet, optimizer, loss_function, args.device)
		loss_vals.append(loss_val) 
		torch.save({
			'model_state_dict': unet.state_dict(),
			'optim_state_dict': optimizer.state_dict(),
			'epoch': e,
			'loss_values': loss_vals
		}, args.model_path)
		print("Epoch completed and model successfully saved!")

if __name__ == '__main__':
	main()