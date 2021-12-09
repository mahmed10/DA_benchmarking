import torch
import numpy as np
SMOOTH = 1e-6
def iou_calculation(batch, model, device):
	X, labels = batch
	X, labels = X.to(device), labels.to(device)
	pred_main = model(X)[1]
	output = pred_main.cpu().data[0].numpy()
	output = output.transpose(1, 2, 0)
	pred = np.argmax(output, axis=2)

	outputs = torch.tensor(pred.reshape((1, pred.shape[0], pred.shape[1]))).cpu()
	labels = torch.tensor(labels).cpu()
	# print(outputs.size())
	# print(labels.size())
	intersection = (outputs & labels).float().sum((1, 2)) 
	union = (outputs | labels).float().sum((1, 2)) 
	iou = (intersection + SMOOTH) / (union + SMOOTH)
	# print(iou)
	# outputs = model(X)[1]
	# outputs = torch.nn.functional.softmax(outputs, dim=1)
	# outputs = torch.argmax(outputs, dim=1)
	# outputs = torch.tensor(outputs.reshape((1, outputs.shape[0], outputs.shape[1])))
	# labels = torch.tensor(labels)
	# intersection = (outputs & labels).float().sum((1, 2)) 
	# union = (outputs | labels).float().sum((1, 2)) 
	# iou = iou + (intersection + SMOOTH) / (union + SMOOTH)
	#iou = torch.clamp(20 * (iou - 0.5), 0, 10) / 10
	return iou