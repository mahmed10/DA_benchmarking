from data import relis3d


def config_loader(dataset, train, path_list, data_mode):
	if dataset == 'relis3d':
		return relis3d.Relis3D(train, path_list, data_mode)