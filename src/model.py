import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context  # for loading ResNet-18

from config import config

class RhythmNet(nn.Module):
    """ RhythmNet model architecture. """
    def __init__(self):
        super(RhythmNet, self).__init__()
        # CNN model
        resnet = models.resnet18(weights='DEFAULT')
        modules = list(resnet.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)     # remove the classifier layer of ResNet-18
        
        # GRU model
        self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=1)

        # linear layer (after CNN)
        self.linear = nn.Linear(512, 1000)

        # regression layer (after both CNN and GRU)
        self.regression = nn.Linear(1000, 1)

        # regularisation
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(0.3)

    def forward(self, st_maps):
        """ Forward pass through RhythmNet.
        
        Args:
            st_maps (torch.Tensor): 4D tensor of stacked ST maps of all clips in batch.
        """
        cnn_frame_preds = []
        cnn_frame_feats = []
        gru_frame_preds = []

        # CNN processing
        for t in range(st_maps.size(1)):                                    # st_maps shape: (B, T, n, c)
            # pass ST maps of all clips in batch at frame t through ResNet-18
            x = (st_maps[:, t, :, :].permute(0, 2, 1)).unsqueeze(-1)        # shape: (B, c, n, 1)
            x = self.resnet18(x)                                            # shape: (B, 512, 1, 1)

            # collapse frame-wise CNN features to 2d tensor
            x = x.view(x.size(0), -1)                                       # shape: (B, 512)

            # regularisation and linear projection of frame-wise CNN features
            x = self.batch_norm1(x)
            x = self.dropout(x)
            x = F.relu(self.linear(x))                                      # shape: (B, 1000)
            x = self.batch_norm2(x)
            x = self.dropout(x)
            cnn_frame_feats.append(x)

            # get CNN's frame-wise HR predictions
            x = self.regression(x)                                          # shape: (B, 1)
            x = x * config.FRAME_RATE
            cnn_frame_preds.append(x.squeeze(-1))

        # GRU processing
        # compute CNN's clip-wise HR predictions by averaging its frame-wise predictions across all frames in clip
        cnn_clip_preds = torch.stack(cnn_frame_preds, dim=0).mean(dim=0)    # shape after stacking: (T, B), shape after mean: (B,)

        # pass frame-wise CNN features through GRU
        gru_frame_feats, _ = self.rnn(torch.stack(cnn_frame_feats, dim=0))  # shape before and after GRU: (T, B, 1000)
        
        # get GRU's frame-wise HR predictions
        for i in range(gru_frame_feats.size(0)):
            y = self.regression(gru_frame_feats[i, :, :])                   # shape: (B, 1)
            gru_frame_preds.append(y.squeeze(-1))

        # compute GRU's clip-wise HR predictions by averaging its frame-wise predictions across all frames in clip
        gru_clip_preds = torch.stack(gru_frame_preds, dim=0).mean(dim=0)    # shape after stacking: (T, B), shape after mean: (B,)
        
        # return all clip-wise HR predictions from CNN and only those clip-wise HR predictions from GRU required for smooth loss
        return cnn_clip_preds, gru_clip_preds[:config.SMOOTH_LOSS_WINDOW]   # shape: (B,), (6,)