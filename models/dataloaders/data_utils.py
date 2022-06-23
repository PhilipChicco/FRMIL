from common.utils import set_seed


def dataset_builder(args):
    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'cm16':
        from models.dataloaders.wsi_datasets import WSIBagDataset as Dataset
    elif args.dataset == 'simclr':
        from models.dataloaders.wsi_datasets import WSIFolders as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset