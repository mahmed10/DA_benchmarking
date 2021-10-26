from data import relis3d
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader

def setup_loaders(args):
	args.dataset_cls = relis3d

	mean_std = ([0.496588, 0.59493099, 0.53358843], [0.496588, 0.59493099, 0.53358843])

	args.num_workers = 4

	train_joint_transform_list = [joint_transforms.RandomSizeAndCrop(args.crop_size, False,
																pre_size=args.pre_size,
																scale_min=args.scale_min,
																scale_max=args.scale_max),
									joint_transforms.Resize(args.crop_size), 
									joint_transforms.RandomHorizontallyFlip()]

	train_joint_transform = joint_transforms.Compose(train_joint_transform_list)
	train_input_transform = []

	train_input_transform += [extended_transforms.ColorJitter(
		brightness=args.color_aug,
		contrast=args.color_aug,
		saturation=args.color_aug,
		hue=args.color_aug)]

	train_input_transform += [extended_transforms.RandomGaussianBlur()]

	train_input_transform += [standard_transforms.ToTensor(),
		standard_transforms.Normalize(*mean_std)]
	
	train_input_transform = standard_transforms.Compose(train_input_transform)

	target_transform = extended_transforms.MaskToTensor()

	target_train_transform = extended_transforms.MaskToTensor()

	train_set = args.dataset_cls.Relis3D(
                args.path_list, 
                joint_transform=train_joint_transform,
                transform=train_input_transform,
                target_transform=target_train_transform,
                dump_images=args.dump_augmentation_images,
                cv_split=args.cv)

	train_sampler = None

	train_loader = DataLoader(train_set, batch_size=args.train_batch_size)

	return train_loader