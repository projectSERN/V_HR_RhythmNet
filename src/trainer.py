import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import config
from model import RhythmNet
from utils.rhythmnet_loss import RhythmNetLoss
from utils.metrics import *
from utils.plots import *

class RhythmNetTrainer:
    """ Trainer class for RhythmNet model training and evaluation.
    
    Attributes:
        train_dl (torch.utils.data.DataLoader): DataLoader for training set.
        val_dl (torch.utils.data.DataLoader): DataLoader for validation set.
        test_dl (torch.utils.data.DataLoader): DataLoader for test set.
        model (torch.nn.Module): RhythmNet model.
        loss_fn (torch.nn.Module): RhythmNet loss function (CNN/L1 loss + GRU/smooth loss).
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
        epoch (int): Current epoch number.
        patience (int): Patience counter for early stopping.
        best_val_mae (float): Best validation MAE value for saving best model and early stopping.
        train_val_stats (dict): Dictionary to store training and validation metrics and losses per epoch.
        test_results (dict): Dictionary to store test metrics.
    """
    def __init__(self, train_dl, val_dl, test_dl):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.model = RhythmNet().to(config.DEVICE)
        self.loss_fn = RhythmNetLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.8, patience=5, mode='min')
        
        self.epoch = 0
        self.patience = 0
        self.best_val_mae = 0
        self.train_val_stats = {'epoch': [], 'train_mae': [], 'train_rmse': [], 'train_mape': [], 'train_loss': [],
                                'val_mae': [], 'val_rmse': [], 'val_mape': [], 'val_loss': []}
        self.test_results = {}

    def train_epoch(self):
        """ Train the model for one epoch on train set. """
        preds_list = []                                     # len: config.CLIP_SIZE * config.BATCH_SIZE
        targets_list = []                                   # len: config.CLIP_SIZE * config.BATCH_SIZE
        file_paths = []                                     # len: config.CLIP_SIZE * config.BATCH_SIZE
        epoch_loss = 0

        for batch in self.train_dl:
            # load batch data
            st_maps = batch["st_map"].to(config.DEVICE)
            targets = batch["target"].to(config.DEVICE)
            file_paths.extend(batch["file_path"])
            
            # forward pass
            cnn_preds, gru_preds = self.model(st_maps)      # shapes: (B,) and (6,)
            loss = self.loss_fn(cnn_preds, gru_preds, targets)
            epoch_loss += loss.item()

            # backprop and lr step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # extend with predictions and targets of all clips in batch
            preds_list.extend(cnn_preds.detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

        # epoch end: group clips into videos, compute epoch metrics and update self.train_val_stats
        epoch_metrics = compute_metrics(clips_to_videos(preds_list, targets_list, file_paths))
        for metric in ['mae', 'rmse', 'mape']:
            self.train_val_stats[f'train_{metric}'].append(epoch_metrics[f'overall_{metric}'])
        self.train_val_stats['train_loss'].append(epoch_loss / len(self.train_dl))
        self.train_val_stats['epoch'].append(self.epoch)

    def val_epoch(self):
        """ Validate the model for one epoch on val set. """
        preds_list = []                                     # len: config.CLIP_SIZE * config.BATCH_SIZE
        targets_list = []                                   # len: config.CLIP_SIZE * config.BATCH_SIZE
        file_paths = []                                     # len: config.CLIP_SIZE * config.BATCH_SIZE
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.val_dl:
                # load batch data
                st_maps = batch["st_map"].to(config.DEVICE)
                targets = batch["target"].to(config.DEVICE)
                file_paths.extend(batch["file_path"])

                # forward pass
                cnn_preds, gru_preds = self.model(st_maps)  # shapes: (B,) and (6,)
                loss = self.loss_fn(cnn_preds, gru_preds, targets)
                epoch_loss += loss.item()

                # extend with predictions and targets of all clips in batch
                preds_list.extend(cnn_preds.detach().cpu().numpy())
                targets_list.extend(targets.detach().cpu().numpy())

        # epoch end: group clips into videos, compute epoch metrics and update self.train_val_stats
        epoch_metrics = compute_metrics(clips_to_videos(preds_list, targets_list, file_paths))
        for metric in ['mae', 'rmse', 'mape']:
            self.train_val_stats[f'val_{metric}'].append(epoch_metrics[f'overall_{metric}'])
        self.train_val_stats['val_loss'].append(epoch_loss / len(self.val_dl))

    def test_epoch(self):
        """ Test the model on the test set. """
        preds_list = []                                     # len: config.CLIP_SIZE * config.BATCH_SIZE
        targets_list = []                                   # len: config.CLIP_SIZE * config.BATCH_SIZE
        file_paths = []                                     # len: config.CLIP_SIZE * config.BATCH_SIZE
        epoch_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.test_dl, total=len(self.test_dl), desc="Batches"):
                # load data from batch
                st_maps = batch["st_map"].to(config.DEVICE)
                targets = batch["target"].to(config.DEVICE)
                file_paths.extend(batch["file_path"])

                # forward pass
                cnn_preds, gru_preds = self.model(st_maps)  # shapes: (B,) and (6,)
                loss = self.loss_fn(cnn_preds, gru_preds, targets)
                epoch_loss += loss.item()

                # extend with predictions and targets of all clips in batch
                preds_list.extend(cnn_preds.detach().cpu().numpy())
                targets_list.extend(targets.detach().cpu().numpy())

        # epoch end: group clips into videos and compute metrics over test set
        self.test_results = compute_metrics(clips_to_videos(preds_list, targets_list, file_paths))

    def train(self):
        """ Training routine for RhythmNet model. """
        pbar = tqdm(range(1, config.EPOCHS + 1), desc="Epochs")
        for self.epoch in pbar:
            # training epoch
            self.model.train()
            self.train_epoch()

            # validation epoch
            self.model.eval()
            self.val_epoch()

            # update learning rate
            self.scheduler.step(self.train_val_stats['val_loss'][-1])

            # save checkpoint if validation MAE improves
            if len(self.train_val_stats['val_mae']) == 1:
                self.best_val_mae = self.train_val_stats['val_mae'][-1]
                self.save_checkpoint(mode='best')
                self.patience = 0
            else:
                if self.best_val_mae > self.train_val_stats['val_mae'][-1]:
                    self.best_val_mae = self.train_val_stats['val_mae'][-1]
                    self.save_checkpoint(mode='best')
                    self.patience = 0
                else:
                    self.patience += 1

            # early stopping
            if self.patience >= config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {self.epoch}")
                break
            
            # update progress bar with this validation epoch's loss and overall MAE
            pbar.set_postfix({
                'val_loss': self.train_val_stats['val_loss'][-1],
                'val_mae': self.train_val_stats['val_mae'][-1],
            })

        self.save_checkpoint(mode='last')

        # plot and save training and validation metrics per epoch
        plot_train_val_stats(self.train_val_stats)
        save_stats(self.train_val_stats, mode='train_val')

    def test(self):
        """ Testing routine for RhythmNet model. """
        # load best model, overwriting current model
        self.load_model()
        
        # test epoch
        self.model.eval()
        self.test_epoch()

        # save and print test results
        save_stats(self.test_results, mode='test')
        print(f"\nOverall MAE: {self.test_results['overall_mae']:.4g}")
        print(f"Overall RMSE: {self.test_results['overall_rmse']:.4g}")
        print(f"Overall MAPE: {self.test_results['overall_mape']:.4g}%")
        print(f"Number of videos: {self.test_results['num_videos']}")
        print("----")

        # optionally plot bland altman and GT vs pred plots for a specific video in test set
        # if config.PLOT_VIDEO is not None:
        #     plot_gt_vs_pred(..., subject_id=config.PLOT_VIDEO)
        #     plot_bland_altman(..., subject_id=config.PLOT_VIDEO)
    
    def save_checkpoint(self, mode):
        """ Save model and optimiser to a checkpoint file.

        Args:
            mode (str): 'best' or 'last' checkpoint to save.
        """
        checkpoint_path = os.path.join(config.RESULTS_PATH, f"{mode}_model.pth")
        checkpoint = {
            'model': self.model.state_dict(),
            'optimiser': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_model(self):
        """ Load model and optimiser from checkpoint file. """
        checkpoint_path = os.path.join(config.RESULTS_PATH, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            print("\nNo checkpoint file found. Train model first.")
        else:
            if config.DEVICE == 'cpu':
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimiser'])