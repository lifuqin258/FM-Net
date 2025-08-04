import torchvision.models as models
import torch
import torch.nn as nn
from Encoder import *
from Decoder_alex_no import *
from Aggregation import *


class FM(nn.Module):
    def __init__(self, num_classes=6):
        super(FM, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.eval()
        self.feature_extractor_x = alexnet.features
        self.feature_extractor_y = alexnet.features

        self.decoder = Decoder(dim=256)

        self.aggregation = Aggregation()

        self.focal_x = FocalM(dim=256)
        self.focal_y = FocalM(dim=256)
        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x, y):
        x = self.feature_extractor_x(x)
        y = self.feature_extractor_y(y)

        q = self.decoder(x, y)

        x_focal = self.focal_x(x, q)
        y_focal = self.focal_y(y, q)

        aggregation = self.aggregation(x_focal, y_focal)


        aggregation = self.dropout(aggregation)
        aggregation = aggregation.reshape(aggregation.size(0), -1)
        aggregation = self.fc(aggregation)
        return aggregation
