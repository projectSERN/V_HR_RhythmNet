import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VIPL_HR_V1(Dataset):
    """ Dataset class for VIPL-HR-V1 train / val / test set videos (split into clips here).

        Attributes:
            st_map_paths (list or dict): Paths to .npy ST maps of all videos in train/val/test set.
            stride (int): Stride value between clips, in frames.
            st_maps (list): List of (path to clip's .npy ST map, its index in this ST map) tuples.
            targets (list): List of clip-wise ground truth HR values (average across all frames in clip).
    """
    def __init__(self, st_map_paths, stride):
        self.st_map_paths = st_map_paths
        self.stride = stride
        self.st_maps = []                                                           # len: number of clips in train / val / test set
        self.targets = []                                                           # len: number of clips in train / val / test set

        for st_map_path in st_map_paths:                                            # iterate over all .npy ST maps in train / val / test set
            try:
                num_clips = len(np.load(st_map_path, mmap_mode='r'))                # number of clips this ST map's video is split into
                clip_gt_paths = glob.glob(os.path.join(os.path.dirname(st_map_path), 'gt_HRs', 'gt_HR_clip_*.txt'))

                for clip_gt_path in clip_gt_paths:
                    clip_idx = int(os.path.basename(clip_gt_path).split("_")[-1].replace(".txt", ""))
                    assert 0 <= clip_idx < num_clips, f"Ground truth for clip {clip_idx} out of bounds for {st_map_path}."

                    clip_gt = self.get_clip_gt(clip_gt_path)
                    if clip_gt is not None:                                         # any clips with invalid ground truth data are skipped
                        self.st_maps.append((st_map_path, clip_idx))
                        self.targets.append(clip_gt)

            except Exception as e:
                print(f"Error processing a clip in {st_map_path}: {str(e)}")
                continue

    def get_clip_gt(self, clip_gt_path):
        """ Compute and return the clip's average HR value from its ground truth file.
        
        Args:
            clip_gt_path (str): Path to the clip's ground truth HR values text file.
        """
        try:
            if not os.path.exists(clip_gt_path):
                print(f"Ground truth file {clip_gt_path} does not exist.")
                return None

            clip_gt = pd.read_csv(clip_gt_path, header=None).values.flatten()
            clip_gt = np.array(clip_gt, dtype=float)

            if len(clip_gt) == 0:
                print(f"Ground truth file {clip_gt_path} is empty.")
                return None
            elif np.any(clip_gt > 300) or np.any(clip_gt < 30):
                print(f"Ground truth file {clip_gt_path} contains invalid HR values.")
                return None
            
            return np.mean(clip_gt)
        
        except Exception as e:
            print(f"Error processing ground truth file {clip_gt_path}: {str(e)}")
            return None
    
    def __len__(self):
        """ Return the number of clips in the train / val / test set. """
        return len(self.st_maps)

    def __getitem__(self, index):
        """ Return a clip's ST map and ground truth HR value (average across all frames in clip).
        
        Args:
            index (int): Index of the clip in the train / val / test set.
        """

        # load only the clip's ST map, not ST maps from all clips in the video
        st_map_path, clip_idx = self.st_maps[index]
        st_maps_file = np.load(st_map_path, mmap_mode='r')
        st_map = st_maps_file[clip_idx].copy()
        del st_maps_file

        st_map = torch.from_numpy(st_map).float()
        target_hr = torch.tensor(self.targets[index], dtype=torch.float)

        return {
            "st_map": st_map,           # 3D tensor of shape: (T, n, c)
            "target": target_hr,        # 1D tensor of shape: (1,)
            "file_path": st_map_path    # config.ST_MAPS_PATH/p*/v*/source1/st_maps.npy of this clip
        }