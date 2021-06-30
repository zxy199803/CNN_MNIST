import time
from datetime import timedelta
import torchvision
from torch.utils.data import DataLoader


def load_data_mnist(batch_size, resize=None, root='.\data'):
    trains = []
    if resize:
        trains.append(torchvision.transforms.Resize(size=resize))
    trains.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trains)
    minist_train = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=transform)
    minist_test = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transform)

    train_iter = DataLoader(minist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(minist_test, batch_size=batch_size, shuffle=True)

    return train_iter, test_iter


def get_time_dif(start_time):
    """获取已使用的时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def show_and_write_has_dev(writer, total_batch, train_loss, train_acc, time_dif, dev_loss, dev_acc,
                           improve):
    # 打印信息
    msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2%}, Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  ' \
          'Time: {5} {6} '
    print(msg.format(total_batch, train_loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
    # 写日志
    writer.add_scalar("loss/train", train_loss.item(), total_batch)
    writer.add_scalar("loss/dev", dev_loss, total_batch)
    writer.add_scalar("acc/train", train_acc, total_batch)
    writer.add_scalar("acc/dev", dev_acc, total_batch)


def show_and_write(writer, total_batch, train_loss, train_acc, time_dif):
    msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2%}, Time: {3}'
    print(msg.format(total_batch, train_loss.item(), train_acc, time_dif))
    # 写日志
    writer.add_scalar("loss/train", train_loss.item(), total_batch)
    writer.add_scalar("acc/train", train_acc, total_batch)


def plt_confusion_matrix(test_confussion):
    """可视化混淆矩阵"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    label_txt = [str(i) for i in range(10)]
    fig, ax = plt.subplots()
    sns.heatmap(test_confussion, ax=ax, cmap="Blues", vmax=10, cbar=False, annot=True, fmt="d")
    ax.set_xticklabels(label_txt, rotation=0, horizontalalignment='left', family='Times New Roman', fontsize=10)
    ax.set_yticklabels(label_txt, rotation=0, family='Times New Roman', fontsize=10)
    ax.xaxis.set_ticks_position("top")
    plt.show()
