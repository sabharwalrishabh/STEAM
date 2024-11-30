from new_utils import get_imagenet100_loader, get_cifar10_loader, get_imagenet1k_loader

def get_dataset(data_name, data_path):
    train_dataset, test_dataset = None, None
    if data_name == "imagenet100":
        train_dataset, test_dataset = get_imagenet100_loader(data_path)
    elif data_name == "cifar10":
        train_dataset, test_dataset = get_cifar10_loader(data_path)
    elif data_name == "imagenet1k":
        train_dataset, test_dataset = get_imagenet1k_loader(data_path)
    else:
        raise Exception("Wrong data_name provided")
    return train_dataset, test_dataset
