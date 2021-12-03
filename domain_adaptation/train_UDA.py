import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from domain_adaptation.discriminator import get_fc_discriminator
from domain_adaptation.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from domain_adaptation.utils.func import loss_calc, bce_loss
from domain_adaptation.utils.loss import entropy_loss
from domain_adaptation.utils.func import prob_2_entropy
from domain_adaptation.utils.viz_segmask import colorize_mask

def train_advent(model, trainloader, targetloader, model_path):
	''' UDA training with advent
	'''
	# Create the model and start the training.
	input_size_source = (572,572)
	input_size_target = (572,572)
	device = 'cuda:0'
	num_classes = 10

	# SEGMNETATION NETWORK
	model.train()
	model.to(device)
	cudnn.benchmark = True
	cudnn.enabled = True

	# DISCRIMINATOR NETWORK
	# feature-level
	d_aux = get_fc_discriminator(num_classes=num_classes)
	d_aux.train()
	d_aux.to(device)

	# seg maps, i.e. output, level
	d_main = get_fc_discriminator(num_classes=num_classes)
	d_main.train()
	d_main.to(device)

	# OPTIMIZERS
	# segnet's optimizer
	optimizer = optim.SGD(model.parameters(),
						lr=2.5e-4,
						momentum=0.9,
						weight_decay=0.00005)

	# discriminators' optimizers
	optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=1e-4, betas=(0.9, 0.99))
	optimizer_d_main = optim.Adam(d_main.parameters(), lr=1e-4, betas=(0.9, 0.99))

	# interpolate output segmaps
	interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
	interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

	# labels for adversarial training
	source_label = 0
	target_label = 1
	# import pdb; pdb.set_trace()
	for e_epoch in tqdm(range(30)):
		trainloader_iter = enumerate(trainloader)
		targetloader_iter = enumerate(targetloader)
		for i_iter in tqdm(range(200)):
		# reset optimizers
			optimizer.zero_grad()
			optimizer_d_aux.zero_grad()
			optimizer_d_main.zero_grad()
			# adapt LR if needed
			adjust_learning_rate(optimizer, i_iter)
			adjust_learning_rate_discriminator(optimizer_d_aux, i_iter)
			adjust_learning_rate_discriminator(optimizer_d_main, i_iter)

			# UDA Training
			# only train segnet. Don't accumulate grads in disciminators
			for param in d_aux.parameters():
				param.requires_grad = False
			for param in d_main.parameters():
				param.requires_grad = False
			# train on source
			_, batch = trainloader_iter.__next__()
			images_source, labels = batch
			pred_src_aux, pred_src_main = model(images_source.cuda(device))
			pred_src_aux = interp(pred_src_aux)
			loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
			pred_src_main = interp(pred_src_main)
			loss_seg_src_main = loss_calc(pred_src_main, labels, device)
			loss = (1.0 * loss_seg_src_main + 0.1 * loss_seg_src_aux)
			loss.backward()

			# adversarial training ot fool the discriminator
			_, batch = targetloader_iter.__next__()
			images, _= batch
			pred_trg_aux, pred_trg_main = model(images.cuda(device))
			pred_trg_aux = interp_target(pred_trg_aux)
			d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
			loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
			pred_trg_main = interp_target(pred_trg_main)
			d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
			loss_adv_trg_main = bce_loss(d_out_main, source_label)
			loss = (0.001 * loss_adv_trg_main + 0.0002 * loss_adv_trg_aux)
			loss = loss
			loss.backward()

			# Train discriminator networks
			# enable training mode on discriminator networks
			for param in d_aux.parameters():
				param.requires_grad = True
			for param in d_main.parameters():
				param.requires_grad = True
			# train with source
			pred_src_aux = pred_src_aux.detach()
			d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
			loss_d_aux = bce_loss(d_out_aux, source_label)
			loss_d_aux = loss_d_aux / 2
			loss_d_aux.backward()
			pred_src_main = pred_src_main.detach()
			d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
			loss_d_main = bce_loss(d_out_main, source_label)
			loss_d_main = loss_d_main / 2
			loss_d_main.backward()

			# train with target
			pred_trg_aux = pred_trg_aux.detach()
			d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
			loss_d_aux = bce_loss(d_out_aux, target_label)
			loss_d_aux = loss_d_aux / 2
			loss_d_aux.backward()
			pred_trg_main = pred_trg_main.detach()
			d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
			loss_d_main = bce_loss(d_out_main, target_label)
			loss_d_main = loss_d_main / 2
			loss_d_main.backward()

			optimizer.step()
			optimizer_d_aux.step()
			optimizer_d_main.step()

			current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
							'loss_seg_src_main': loss_seg_src_main,
							'loss_adv_trg_aux': loss_adv_trg_aux,
							'loss_adv_trg_main': loss_adv_trg_main,
							'loss_d_aux': loss_d_aux,
							'loss_d_main': loss_d_main}
			print_losses(current_losses, i_iter)
			torch.save({
				'model_state_dict': model.state_dict(),
				'optim_state_dict': optimizer.state_dict(),
				'epoch': e_epoch,
				'iteration': i_iter,
				'loss_seg_src_aux': loss_seg_src_aux,
				'loss_seg_src_main': loss_seg_src_main,
				'loss_adv_trg_aux': loss_adv_trg_aux,
				'loss_adv_trg_main': loss_adv_trg_main,
				'loss_d_aux': loss_d_aux,
				'loss_d_main': loss_d_main
			}, model_path + 'e_'+str(e_epoch).zfill(4)+ 'i_'+str(i_iter).zfill(4))
			sys.stdout.flush()

def print_losses(current_losses, i_iter):
	list_strings = []
	for loss_name, loss_value in current_losses.items():
		list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
	full_string = ' '.join(list_strings)
	tqdm.write(f'iter = {i_iter} {full_string}')



def to_numpy(tensor):
	if isinstance(tensor, (int, float)):
		return tensor
	else:
		return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, model_path):
	print(model_path)
	train_advent(model, trainloader, targetloader, model_path)