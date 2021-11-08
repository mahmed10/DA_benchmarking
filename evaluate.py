import torch
SMOOTH = 1e-6
def iou_calculation(data, model, device):
	iou = 0
	for index, batch in enumerate(data): 
		X, labels = batch
		X, labels = X.to(device), labels.to(device)
		outputs = model(X)
		outputs = torch.nn.functional.softmax(outputs, dim=1)
		outputs = torch.argmax(outputs, dim=1)
		intersection = (outputs & labels).float().sum((1, 2)) 
		union = (outputs | labels).float().sum((1, 2)) 
		iou = iou + (intersection + SMOOTH) / (union + SMOOTH)
		#iou = torch.clamp(20 * (iou - 0.5), 0, 10) / 10
	return iou/(index+1)