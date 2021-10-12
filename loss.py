import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

def get_loss(args):
	criterion = JointEdgeSegLoss(classes= 10, ignore_index=0, upper_bound=args.wt_bound,
		edge_weight=args.edge_weight, seg_weight=args.seg_weight, att_weight=args.att_weight, dual_weight=args.dual_weight).cuda()
	return criterion

class JointEdgeSegLoss(nn.Module):
	def __init__(self, classes, weight=None, reduction='mean', ignore_index=255, norm=False, upper_bound=1.0, mode='train',
		edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
		
		super(JointEdgeSegLoss, self).__init__()

		self.num_classes = classes
		self.seg_loss = ImageBasedCrossEntropyLoss2d(classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()

		self.ignore_index = ignore_index

		self.edge_weight = edge_weight
		self.seg_weight = seg_weight
		self.att_weight = att_weight
		self.dual_weight = dual_weight

	def bce2d(self, input, target):
		n, c, h, w = input.size()

		log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
		target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
		target_trans = target_t.clone()

		pos_index = (target_t ==1)
		neg_index = (target_t ==0)
		ignore_index=(target_t >1)

		target_trans[pos_index] = 1
		target_trans[neg_index] = 0

		pos_index = pos_index.data.cpu().numpy().astype(bool)
		neg_index = neg_index.data.cpu().numpy().astype(bool)
		ignore_index=ignore_index.data.cpu().numpy().astype(bool)

		weight = torch.Tensor(log_p.size()).fill_(0)
		weight = weight.numpy()
		pos_num = pos_index.sum()
		neg_num = neg_index.sum()
		sum_num = pos_num + neg_num
		weight[pos_index] = neg_num*1.0 / sum_num
		weight[neg_index] = pos_num*1.0 / sum_num

		weight[ignore_index] = 0

		weight = torch.from_numpy(weight)
		weight = weight.cuda()
		loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
		return loss

	def forward(self, inputs, targets):
		segin, edgein = inputs
		segmask, edgemask = targets
		losses = {}

		losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
		losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edgein, edgemask)
		losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
		losses['dual_loss'] = self.dual_weight * 0

		return losses

	def edge_attention(self, input, target, edge):
		n, c, h, w = input.size()
		filler = torch.ones_like(target) * self.ignore_index
		return self.seg_loss(input,torch.where(edge.max(1)[0] > 0.8, target, filler))


class ImageBasedCrossEntropyLoss2d(nn.Module):
	def __init__(self, classes, weight=None, size_average=True, ignore_index=255, norm=False, upper_bound=1.0):
		super(ImageBasedCrossEntropyLoss2d, self).__init__()
		print("Using Per Image based weighted loss")

		self.num_classes = classes
		self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
		self.norm = norm
		self.upper_bound = upper_bound
		self.batch_weights = False

	def calculateWeights(self, target):
		hist = np.histogram(target.flatten(), range(
			self.num_classes + 1), normed=True)[0]
		if self.norm:
			hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
		else:
			hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
		return hist

	def forward(self, inputs, targets):
		target_cpu = targets.data.cpu().numpy()
		#print("loss",np.unique(target_cpu))
		if self.batch_weights:
			weights = self.calculateWeights(target_cpu)
			self.nll_loss.weight = torch.Tensor(weights).cuda()

		loss = 0.0
		for i in range(0, inputs.shape[0]):
			if not self.batch_weights:
				weights = self.calculateWeights(target_cpu[i])
				self.nll_loss.weight = torch.Tensor(weights).cuda()
			loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),targets[i].unsqueeze(0))
		return loss