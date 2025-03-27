import os
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import config
from dataset import VIPL_HR_V1
from trainer import RhythmNetTrainer

def main():
    """ Training and evaluation routines for RhythmNet model. """

    ## Set up environment ##
    # GPU and CUDA settings
    if config.GPU >= 0:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU)
        torch.cuda.empty_cache()

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # random seed settings
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # results directory
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    ## Load data ##
    # extract ST maps of all video samples in the dataset
    all_st_map_files = glob.glob(os.path.join(config.ST_MAPS_PATH, 'p*', 'v*', 'source1', 'st_maps.npy'), recursive=True)
    all_st_map_files = all_st_map_files[:(len(all_st_map_files) // config.BATCH_SIZE) * config.BATCH_SIZE]              # truncate to a multiple of batch size

    # split video samples into train (70%), validation (10%) and test (20%) sets
    video_files_train, video_files_test = train_test_split(all_st_map_files, test_size=0.2, random_state=config.SEED)   # 0.2 x 1 = 20% test
    video_files_train, video_files_val = train_test_split(video_files_train, test_size=0.125, random_state=config.SEED) # 0.125 x 0.8 10% val

    print(f"----\nDATASET: {config.DATASET}")
    print(f"Training set: {len(video_files_train)} videos")
    print(f"Val set: {len(video_files_val)} videos")
    print(f"Test set: {len(video_files_test)} videos")

    # split train/val/test set video samples and their ground truth labels into clips and create train/val/test DataLoaders
    train_dl = DataLoader(
        VIPL_HR_V1(video_files_train, config.STRIDE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True, shuffle=False,                                                                                  # shuffle=False for GRU/smooth loss computation
    )

    val_dl = DataLoader(
        VIPL_HR_V1(video_files_val, config.STRIDE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True, shuffle=False,
    )

    test_dl = DataLoader(
        VIPL_HR_V1(video_files_test, config.STRIDE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        drop_last=True, shuffle=False,
    )

    ## Train and evaluate model ##
    trainer = RhythmNetTrainer(train_dl, val_dl, test_dl)

    print(f"----\nTRAINING & VAL")
    with open(f"{config.RESULTS_PATH}/config_args.txt", 'w') as results_file:
        for arg in vars(config):
            results_file.write(f"{arg:<30}: {getattr(config, arg)}\n")
    trainer.train()

    print(f"----\nTESTING")
    trainer.test()

if __name__ == '__main__':
    main()