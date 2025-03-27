import os
import io
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import config

# train and val plots #
def plot_train_val_stats(train_val_stats):
    plots_path = os.path.join(config.RESULTS_PATH, 'train_val_stats')
    os.makedirs(plots_path, exist_ok=True)
    
    metric_configs = [
        ('MAE', 'Mean Absolute Error (bpm)', train_val_stats['train_mae'], train_val_stats['val_mae'], 'b'),
        ('RMSE', 'Root Mean Square Error (bpm)', train_val_stats['train_rmse'], train_val_stats['val_rmse'], 'r'),
        ('MAPE', 'Mean Absolute Percentage Error (%)', train_val_stats['train_mape'], train_val_stats['val_mape'], 'm'),
        ('Loss', 'Loss', train_val_stats['train_loss'], train_val_stats['val_loss'], 'g')
    ]

    # generate and save individual train+val plots for each metric
    for metric_name, y_label, train_metric, val_metric, colour in metric_configs:
        plt.figure(figsize=(10, 6))
        plt.plot(train_val_stats['epoch'], train_metric, f'{colour}-', label=f'Train {metric_name}')
        plt.plot(train_val_stats['epoch'], val_metric, f'{colour}--', label=f'Val {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.title(f'Training and Validation {metric_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_path, f'{metric_name.lower()}.png'))
        plt.close()

    # generate and save a single train+val plot with metrics' subplots
    # MAE subplot
    plt.figure(figsize=(15,10))
    plt.subplot(2, 2, 1)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_mae'], 'b-', label='Train MAE')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_mae'], 'b--', label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (bpm)')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)

    # RMSE subplot
    plt.subplot(2, 2, 2)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_rmse'], 'r-', label='Train RMSE')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_rmse'], 'r--', label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Square Error (bpm)')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.grid(True)

    # MAPE subplot
    plt.subplot(2, 2, 3)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_mape'], 'm-', label='Train MAPE')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_mape'], 'm--', label='Val MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('Training and Validation MAPE')
    plt.legend()
    plt.grid(True)

    # Loss subplot
    plt.subplot(2, 2, 4)
    plt.plot(train_val_stats['epoch'], train_val_stats['train_loss'], 'g-', label='Train Loss')
    plt.plot(train_val_stats['epoch'], train_val_stats['val_loss'], 'g--', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'all_metrics.png'))
    plt.close()

# test plots #
def plot_gt_vs_pred(targets, preds, video_path):
    fig = plt.figure()
    plt.scatter(np.asarray(targets), np.asarray(preds))
    plt.title(f'Ground truth vs predicted HR values per clip for subject {video_path}')
    plt.ylabel('Predicted HR (bpm)')
    plt.xlabel('Ground truth HR (bpm)')

    plot_path = os.path.join(config.RESULTS_PATH, 'GT_vs_pred')
    os.makedirs(plot_path, exist_ok=True)
    fig.savefig(os.path.join(plot_path, f'{video_path.split("/")[-1].split(".")[0]}.png'), dpi=fig.dpi)

def plot_bland_altman(data1, data2, video_path):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    plot_path = os.path.join(config.RESULTS_PATH, 'bland_altman')
    os.makedirs(plot_path, exist_ok=True)
    fig.savefig(os.path.join(plot_path, f'{video_path.split("/")[-1].split(".")[0]}.png'), dpi=fig.dpi)

# ST maps plots #
def plot_st_maps_ubfc(base_path):
    # Get all .npy files
    st_map_files = glob(os.path.join(base_path, "subject*/*.npy"))
    
    # Calculate total number of maps
    total_maps = 0
    for file in st_map_files:
        st_maps = np.load(file)
        total_maps += len(st_maps)
    
    print(f"Found {len(st_map_files)} files with total {total_maps} maps")
    
    # Process all maps with single progress bar
    pbar = tqdm(total=total_maps, desc="Processing all maps")
    
    for st_map_file in st_map_files:
        dir_path = os.path.dirname(st_map_file)
        base_name = os.path.splitext(os.path.basename(st_map_file))[0]
        plots_dir = os.path.join(dir_path, f"{base_name}_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        st_maps = np.load(st_map_file)
        for map_idx, st_map in enumerate(st_maps):
            # Plot channels
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
            
            # Y, U, V channels
            for ax, data, title in zip([ax1, ax2, ax3], 
                                     [st_map[:,:,0], st_map[:,:,1], st_map[:,:,2]], 
                                     ['Y Channel', 'U Channel', 'V Channel']):
                im = ax.imshow(data.T, aspect='auto', cmap='viridis')
                ax.set_title(f'{title} - Map {map_idx+1}')
                ax.set_xlabel('Time (frames)')
                ax.set_ylabel('ROI Blocks')
                plt.colorbar(im, ax=ax).set_label('Pixel Intensity')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"map_{map_idx+1}_channels.png"))
            plt.close()
            
            # Plot combined
            fig, ax = plt.subplots(figsize=(15, 5))
            combined_map = np.mean(st_map, axis=2)
            im = ax.imshow(combined_map.T, aspect='auto', cmap='viridis')
            ax.set_title(f'Combined Channels - Map {map_idx+1}')
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('ROI Blocks')
            plt.colorbar(im).set_label('Average Pixel Intensity')
            
            plt.savefig(os.path.join(plots_dir, f"map_{map_idx+1}_combined.png"))
            plt.close()
            
            pbar.update(1)
    
    pbar.close()

if __name__ == "__main__":
    plot_st_maps_ubfc(config.ST_MAPS_PATH)