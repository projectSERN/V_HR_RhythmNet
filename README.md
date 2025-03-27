# RhythmNet--VIPL_final

## Table of Contents

[Overview](#overview)\
[Code execution](#code-execution)\
[Expected file directories](#expected-file-directories)

## Overview
This repo includes the final implementation of the RhythmNet model, which generates the *V<sub>HR</sub>* modality for the multimodal SERN model.

## Code execution
### Environment setup
1. Clone the repository and navigate to the cloned folder
    ```
    git clone <HTTPS_URL>
    cd RhythmNet--VIPL_final
    ```
2. Create a virtual environment and activate it
3. Install required packages and dependencies
    ```
    pip install -r requirements.txt
    ```
4. Download the `skin_u2netp.onnx` file from [here](https://github.com/samhaswon/skin_segmentation/releases) and place it in `src/u2net/` of this repo folder (do not rename it)
5. Replace all `"/scratch/zce*/"` and `"/home/zce*/"` in `src/config.py` with your own paths to the dataset's folder and repo folder, respectively

### Using VIPL-HR-V1 dataset
1. Ensure dataset's folder is named `VIPL-HR-V1`
2. Generate clip-wise ST maps and ground truth data for all videos in the dataset (with the recommended clip size of 125 frames and stride length of 10 frames)
    ```
    python src/st_maps.py --GPU <#>
    ```
    These are saved to `.../VIPL-HR-V1/st_maps_125c_10s/`, i.e., in the dataset's folder (refer to expected file directories)
3. Train and evaluate the RhythmNet model using these ST maps and ground truth data
    ```
    python src/main.py --GPU <#>
    ```
    All results and plots are saved to `.../RhythmNet--VIPL_final/results_VIPL-HR-V1_125c_10s/`, i.e., in the repo folder (refer to expected file directories)

Additional command-line arguments can be passed  if needed (refer to `src/config.py` for more details on these)

### Using another dataset
1. Generate clip-wise ST maps and ground truth data for all videos in the dataset, using a custom clip size and stride length
    ```
    python src/st_maps.py --GPU <#> --DATASET <xxx> --FRAME_RATE <@> --CLIP_SIZE <*> --STRIDE <$>
    ```
    These are saved to `.../xxx/st_maps_*c_$s/`, i.e., in the dataset's folder
2. Train and evaluate the RhythmNet model using these ST maps and ground truth data
    ```
    python src/main.py --GPU <#> --DATASET <xxx> --FRAME_RATE <@> --CLIP_SIZE <*> --STRIDE <$>
    ```
    All results and plots are saved to `.../RhythmNet--VIPL_final/results_xxx_*c_$s/`, i.e., in the repo folder

## Expected file directories
### Dataset folder
E.g., VIPL-HR-V1 dataset folder (after generating ST maps):
```
/scratch/zce*/VIPL-HR-V1/
|-- p1/
|   |-- v1/
|   |   |-- source1/
|   |   |   |-- video.avi
|   |   |   |-- gt_HR.csv
|   |   |-- ...
|   |   |-- source4/
|   |-- ...
|   |-- v9/
|-- ...
|-- p107/
|
|-- st_maps_125c_10s/
|   |-- p1/
|   |   |-- v1/
|   |   |   |-- source1/
|   |   |   |   |-- st_maps.npy             -> clip-wise ST maps of shape (num of clips in video, 125, 25, 3)
|   |   |   |   |-- gt_HRs/                 -> clip-wise ground truth data
|   |   |   |   |   |-- gt_HR_clip_0.txt
|   |   |   |   |   |-- gt_HR_clip_1.txt
|   |   |   |   |   |-- ...
|   |   |-- ...
|   |   |-- v6/
|   |-- ...
|   |-- p107/
```

### Repo folder
`RhythmNet--VIPL_final` repo (after training and evaluating model using VIPL-HR-V1):
```
/home/zce*/RhythmNet--VIPL_final/
|-- results_125c_10s/
|   |-- best_model.pth
|   |-- config_args.txt
|   |-- last_model.pth
|   |-- test_results.txt
|   |-- train_val_stats/
|       |-- all_metrics.png
|       |-- bland_altman [optional]
|       |-- epoch_stats.txt
|       |-- loss.png
|       |-- mae.png
|       |-- mape.png
|       |-- rmse.png
|       |-- GT_vs_pred [optional]
|
|-- src/
|   |-- rtgene/
|   |   |-- ...
|   |-- u2net/
|   |   |-- inference.py
|   |   |-- skin_u2netp.onnx
|   |-- utils/
|   |   |-- custom_loss.py
|   |   |-- metrics.py
|   |   |-- plots.py
|   |   |-- rhythmnet_loss.py
|   |-- config.py
|   |-- dataset.py
|   |-- main.py
|   |-- model.py
|   |-- st_maps.py
|   |-- trainer.py
|
|-- README.md
|-- requirements.txt
```