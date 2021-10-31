from data import relis3d
from torch.utils.data import DataLoader

def setup_loaders(path_list, batch_size):
	data = relis3d.Relis3D(path_list)
	data_loaded = DataLoader(data, batch_size=batch_size)
	return data_loaded