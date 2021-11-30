import torch
import torch.nn as nn
import torchvision.models as models


class Lrcn(nn.Module):
    def __init__(self):
        super(Lrcn, self).__init__()

        # 定义CNN→这里采用AlexNet，还可以选择其他预训练模型进行实验对比
        # 依据论文https://arxiv.org/abs/1404.5997，移除fc7
        self.feature_extractor = models.alexnet(pretrained=True)
        self.feature_extractor.classifier = nn.Sequential(*list(self.feature_extractor.classifier.children())[:-5])

        # 定义单层lstm
        self.lstm = nn.LSTM(input_size=4096, hidden_size=128, num_layers=1, dropout=0.9, batch_first=True)

        # 定义全连接输出层，in_features=hidden_size
        self.LinearOutput = nn.Linear(in_features=128, out_features=2)

        # 归一化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, frame_sequence):

        # 输入的图片序列→[B, C, T, H, W]
        # lstm所需，即CNN提取出的特征序列→[B, T, input_size]
        feature_sequence = torch.empty(size=(frame_sequence.size()[0], frame_sequence.size()[2], 4096), device='cuda')
        for i in range(0, frame_sequence.size()[1]):
            frame = frame_sequence[:, :, i, :, :]
            feature_frame = self.feature_extractor(frame)
            feature_sequence[:, i, :] = feature_frame

        # 输入lstm
        x, _ = self.lstm(feature_sequence)

        # 输入fc进行分类
        x = self.LinearOutput(x)

        # 平均获取最终结果
        x = torch.mean(x, dim=1)

        return self.softmax(x)


if __name__ == '__main__':
    model = Lrcn().to(torch.device('cuda'))
    frames = torch.rand(10, 3, 30, 227, 227).to(torch.device('cuda'))
    output = model(frames)

    print(output)
