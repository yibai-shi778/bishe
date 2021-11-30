from data_preparation import generator
from models import LRCN
from train_and_test import train
import torch


if __name__ == '__main__':

    # 构造测试及验证数据迭代器
    batch_size = 16
    train_path = 'E:\\yan0\\bishe_test\\Test'
    val_path = 'E:\\yan0\\bishe_test\\Val'
    datatype = 'CNN_input'
    target_shape = [224, 224]

    train_data_iter = generator.seq_data_iter_random(batch_size, train_path, datatype, target_shape)
    val_data_iter = generator.seq_data_iter_random(batch_size, val_path, datatype, target_shape)

    # 构造lrcn模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LRCN.Lrcn().to(device)

    # 训练模型并验证
    num_epochs = 10
    lr = 0.005
    momentum = 0.9

    # 损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 全训练？不用冻住CNN？
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum
    )

    train.train_model_pytorch123(model, train_data_iter, val_data_iter, loss, optimizer, num_epochs)
