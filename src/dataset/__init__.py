from .mnist import MNISTData

_dataset_dict = {
    "mnist": MNISTData
}

def get_dataloader(data_config):
    dataset_name = data_config.dataset_name
    Dataset = _dataset_dict[dataset_name]
    return Dataset.get_dataloader(data_config.dataset_config, data_config.dataloader_config)