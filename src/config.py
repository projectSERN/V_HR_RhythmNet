import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--DATASET', type=str, default='VIPL-HR-V1', help='Dataset to train and evaluate on')
parser.add_argument('--FRAME_RATE', type=float, default=25.0, help='Frame rate of videos')
parser.add_argument('--CLIP_SIZE', type=int, default=125, help='Number of frames per clip')
parser.add_argument('--STRIDE', type=int, default=10, help='Stride length in frames')
parser.add_argument('--SMOOTH_LOSS_WINDOW', type=int, default=6, help='Window size for smooth loss')
parser.add_argument('--ROI_GRID_SIZE', type=int, default=5, help='Side length of square ROI grid for ST map generation')
parser.add_argument('--SKIN_SEG_MODEL', type=str, default='skin_u2netp.onnx', help='Skin segmentation model to use in ST map generation')

parser.add_argument('--LR', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size')
parser.add_argument('--NUM_WORKERS', type=int, default=8, help='Number of workers for data loader')
parser.add_argument('--EPOCHS', type=int, default=20, help='Number of epochs')
parser.add_argument('--PATIENCE', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--PLOT_VIDEO', type=str, help='Path to video for bland altman and GT vs pred plots')

parser.add_argument('--GPU', type=int, default=0, help='ID of GPU to use')
parser.add_argument('--SEED', type=int, default=42, help='Random seed for reproducibility')

config = parser.parse_args()
config.SOURCE_PATH = f"/scratch/zce*/{config.DATASET}/"
config.ST_MAPS_PATH = f"{config.SOURCE_PATH}st_maps_{config.CLIP_SIZE}c_{config.STRIDE}s/"
config.RESULTS_PATH = f"/home/zce*/RhythmNet--VIPL-HR-V1_final/results_{config.DATASET}_{config.CLIP_SIZE}c_{config.STRIDE}s/"
config.DEVICE = 'cuda' if config.GPU >= 0 else 'cpu'