from model.data import get_dataloader


def one_batch_dataloader(file_path, num_workers = 0):
    dataloader = get_dataloader({
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,})
    for data in dataloader:
        for k in data:
            if isinstance(data[k], torch.Tensor):
                print(k, data[k].shape)
            else:
                print(k, data[k])


if __name__ == "__main__":
    file_path = "/home/didi/Desktop/motion_dataset_prerender/debug"
    one_batch_dataloader(file_path, 0)