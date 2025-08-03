import   进口 torchvision.models as   作为 models
import   进口 torch   进口火炬
import   进口 torch.nn    进口火炬。Nn as   作为 Nnas nn
from Encoder import   进口 *
from Decoder_alex_no import   进口 *
from Aggregation import   进口 *


class   类 FM   调频(nn.Module   模块):
    def __init__(self, num_classes=100):
        super   超级(FM, self).__init__()
        alexnet = models.alexnet(pretrained=True   真正的)
        alexnet.eval()
        self.feature_extractor_x = alexnet.features
        self.feature_extractor_y = alexnet.features

        self.decoder = Decoder   译码器(dim=256)

        self.aggregation = Aggregation()

        self.focal_x = FocalM(dim=256)
        self.focal_y = FocalM(dim=256)
        self.dropout = nn.Dropout   辍学(p=0.6)
        self.fc = nn.Linear   线性(256 * 6 * 6, num_classes)

    def forward   向前(self, x, y):
        x = self.feature_extractor_x(x)
        y = self.feature_extractor_y(y)

        q = self.decoder(x, y)

        x_focal = self.focal_x(x, q)
        y_focal = self.focal_y(y, q)

        aggregation = self.aggregation(x_focal, y_focal)


        aggregation = self.dropout(aggregation)
        aggregation = aggregation.reshape(aggregation.size(0), -1)
        aggregation = self.dropout(aggregation)
        aggregation = self.fc(aggregation)
        return aggregation
