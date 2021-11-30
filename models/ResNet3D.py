import torch
import torch.nn as nn
import torchvision.models as models


class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D, self).__init__()

        self.r3d = models.video.r3d_18(pretrained=True, progress=True)
        self.r3d.fc = nn.Linear(in_features=512, out_features=2)

    def forward(self, sequence):

        return nn.Softmax(self.r3d(sequence))


if __name__ == '__main__':
    model = ResNet3D().to(torch.device('cuda'))
    frames = torch.rand(10, 3, 10, 112, 112).to(torch.device('cuda'))
    output = model(frames)

    print(output)
