'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, num_classes)
        self.record = [2,6,9,13,16,19,22,26,29,32,35,39,42,45,49,52]

    def forward(self, x):
        out = self.features(x)
        # out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

    def update(self):
        self.features = self.features[:self.record[-1]+1]
        self.record = self.record[:-1]

        x = torch.randn((1,3,32,32)).cuda()
        self.classifier = nn.Linear(self.features(x).numel(),10).cuda()

    
    


def test():
    net = VGG('VGG11')
    print(net)
    x = torch.randn(2,3,32,32)
    print('# fucknet parameters:', sum(param.numel() for param in net.parameters()))
    y = net(x)
    print(y.size())

if __name__=='__main__':
    test()
