import time
import copy

from models import LRCN
import torch
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(save_path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_path)


def train_model_pytorch123(model, train_iter, val_iter, criterion, optimizer, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_iter = train_iter
            else:
                data_iter = val_iter

            running_loss = 0.0
            running_acc = 0.0
            batch_count = 0

            # 迭代数据
            for inputs, labels in data_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 参数梯度置零
                optimizer.zero_grad()

                # 前向运算并且仅在训练时跟踪梯度变化
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, dim=1)

                    # 反向传播并且仅在训练时进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计→若数据集的长度无法被batch_size整除，最后一个不足batch_size的batch应当如何处理？
                # 为求简便，这次便直接丢弃
                batch_count += 1
                running_loss += loss.item()
                running_acc += torch.sum(preds == labels)

            epoch_loss = running_loss / batch_count
            epoch_acc = running_acc / (batch_count * inputs.shape[0])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存最优模型
    # model.load_state_dict(best_model_wts)
    # return model, val_acc_history


# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     with torch.no_grad():
#         for X, y in data_iter:
#             net.eval()
#             acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
#             net.train()
#             print(y.shape[0])
#             n += y.shape[0]
#     return acc_sum / n
#
#
# def train_model_limu(net, train_iter, val_iter, optimizer, device, num_epoches):
#     net = net.to(device)
#     loss = torch.nn.CrossEntropyLoss()
#     for epoch in range(num_epoches):
#         train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
#         for X, y in train_iter:
#             X = X.to(device)
#             y = y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#         val_acc = evaluate_accuracy(val_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
#               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, val_acc, time.time() - start))
