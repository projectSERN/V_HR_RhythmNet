import os
import numpy as np

from config import config

def clips_to_videos(preds_list, targets_list, file_paths):
    """ Groups clips by the video they belong to, using paths to clips' .npy ST maps.

    Args:
        preds_list (list): List of clip-wise HR predictions from CNN for all clips in epoch.
        targets_list (list): List of clip-wise ground truth HR values for all clips in epoch.
        file_paths (list): List of paths to .npy ST maps for all clips in epoch.

    Returns:
        video_data (dict): Dictionary mapping videos to their clip-wise predictions and ground truths.
            - Key: Path to directory containing clips of a video.
            - Value: Dictionary with keys 'preds' and 'targets' containing lists of clip-wise predictions and ground truths.
    """
    video_data = {}
    for (preds, targets, file_path) in zip(preds_list, targets_list, file_paths):
        video = os.path.dirname(file_path)                              # example: config.ST_MAPS_PATH/p*/v*/source1

        if video not in video_data:
            video_data[video] = {'preds': [], 'targets': []}

        video_data[video]['preds'].append(preds)
        video_data[video]['targets'].append(targets)

    return video_data

def compute_metrics(video_data):
        """ Calculate metrics at video level and average across all videos.

        Args:
            video_data (dict): Dictionary containing clip-wise HR predictions from CNN and ground truth HR values for all clips in epoch.

        Returns:
            epoch_metrics (dict): MAE, RMSE, MAPE averaged across all videos in epoch, and number of videos in epoch.
        """
        # iterate over all videos in epoch and calculate metrics for each
        video_metrics = {}
        for video, video_dict in video_data.items():
            video_preds = np.array(video_dict['preds'])
            video_targets = np.array(video_dict['targets'])

            # calculate video's MAE = absolute difference between predicted and ground truth HRs averaged across all clips in video
            video_mae = np.mean(np.abs(video_preds - video_targets))

            # calculate video's RMSE = sqrt(squared differences between predicted and ground truth HRs averaged across all clips in video)
            video_rmse = np.sqrt(np.mean((video_preds - video_targets) ** 2))

            # calculate MAPE for this video = (MAE / average ground truth HR across all clips in video) * 100
            if np.mean(video_targets) != 0:
                video_mape = (video_mae / np.mean(video_targets)) * 100
            else:
                video_mape = 0

            video_metrics[video] = {
                'mae': video_mae,
                'rmse': video_rmse,
                'mape': video_mape,
                'num_clips': len(video_targets) # not used, just for debugging
            }

        # calculate average metrics across all videos in epoch
        epoch_metrics = {
            "overall_mae": np.mean([video_metric['mae'] for video_metric in video_metrics.values()]),
            "overall_rmse": np.mean([video_metric['rmse'] for video_metric in video_metrics.values()]),
            "overall_mape": np.mean([video_metric['mape'] for video_metric in video_metrics.values()]),
            "num_videos": len(video_metrics)
        }

        return epoch_metrics

def save_stats(stats, mode):
    """ Save training and validation metrics per epoch or test results to a text file.
    
    Args:
        stats (dict): Dictionary containing metrics to save.
        mode (str): 'train_val' or 'test'.
    """
    if mode == 'train_val':
        with open(os.path.join(config.RESULTS_PATH, 'train_val_stats', f'epoch_stats.txt'), 'w') as file:
            file.write('TRAIN AND VAL METRICS PER EPOCH\n\n')
            file.write(f"{'Epoch':<10}{'Train MAE':<15}{'Train RMSE':<15}{'Train MAPE':<15}{'Train Loss':<15}{'Val MAE':<15}{'Val RMSE':<15}{'Val MAPE':<15}{'Val Loss':<15}\n")
            for i in range(len(stats['epoch'])):
                file.write(f"{stats['epoch'][i]:<10}{stats['train_mae'][i]:<15.4g}{stats['train_rmse'][i]:<15.4g}{stats['train_mape'][i]:<15.4g}{stats['train_loss'][i]:<15.4g}{stats['val_mae'][i]:<15.4g}{stats['val_rmse'][i]:<15.4g}{stats['val_mape'][i]:<15.4g}{stats['val_loss'][i]:<15.4g}\n")

    elif mode == 'test':
        with open(os.path.join(config.RESULTS_PATH, f'test_results.txt'), 'w') as file:
            file.write('TEST RESULTS\n\n')
            file.write(f"Overall MAE: {stats['overall_mae']:.4g}\n")
            file.write(f"Overall RMSE: {stats['overall_rmse']:.4g}\n")
            file.write(f"Overall MAPE: {stats['overall_mape']:.4g}%\n")
            file.write(f"Number of videos: {stats['num_videos']}\n")