from importlib import import_module
import torch
import time
from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np
import utils
from utils import get_time_dif, load_data_mnist, show_and_write_has_dev, show_and_write


def trian(model, config, train_iter, test_iter, dev_iter=None):
    print('Train on:', config.device)
    start_time = time.time()
    model.train()  # 仅当采用Batch Normalization和dropout时才有影响
    # optimizer = torch.optim.Adam(model.parameters(), config.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # 学习率指数衰减，每个epoch，学习率=学习率*gamma
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=config.lr_scheduler_gamma)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False  # 记录是否很久没有效果提升

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()) + 'sgd')

    # 训练
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()
        for i, (trians, labels) in enumerate(train_iter):
            # input output
            trians = trians.to(config.device)
            labels = labels.to(config.device)
            outputs = model(trians)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            # backword
            model.zero_grad()  # 清空梯度
            loss.backward()
            optimizer.step()  # 更新参数

            # 验证集上验证
            if config.dev_data and total_batch % config.val_batch == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)  # 每次会覆盖上次结果
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                show_and_write_has_dev(writer, total_batch, loss, train_acc, time_dif, dev_loss, dev_acc, improve)
                model.train()

            # 无验证集时
            else:
                if total_batch % config.val_batch == 0:
                    true = labels.data.cpu()
                    predict = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predict)
                    time_dif = get_time_dif(start_time)
                    show_and_write(writer, total_batch, loss, train_acc, time_dif)

            total_batch += 1
            if config.dev_data and total_batch - last_improve > config.require_improvement:
                print('No optimization for a long time, auto stopping...')
                flag = True
                break
        # 每个epoch在测试集上测试一次
        test_acc, _ = evaluate(config, model, test_iter)
        print('Test acc after ', epoch + 1, ' epoch:', test_acc)
        if flag:
            break

    writer.close()

    # 训练结束后保存模型
    torch.save(model.state_dict(), config.save_path)

    # 训练后进行测试
    test(config, model, test_iter)


def evaluate(config, model, data_iter, is_training=True):
    model.eval()  # 关掉dropout
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():  # 计算不会在反向传播中被记录
        for trains, labels in data_iter:
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            outputs = model(trains)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)

    if not is_training:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion

    return acc, loss_total / len(data_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confussion = evaluate(config, model, test_iter, is_training=False)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print('')
    print('Test Report:')
    print(msg.format(test_loss, test_acc))
    print('Precision, Recal and F1-Score:')
    print(test_report)
    print("Confusion Matrix:")
    print(test_confussion)
    utils.plt_confusion_matrix(test_confussion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


class Run:
    @classmethod
    def load_model(cls, x):
        my_config = x.Config()
        model = x.Model().to(my_config.device)  # 实例化
        return my_config, model

    @classmethod
    def seed(cls):
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    @classmethod
    def train(cls, x):
        my_config, model = Run.load_model(x)
        train_iter, test_iter = load_data_mnist(my_config.batch_size, resize=my_config.resize)  #
        trian(model, my_config, train_iter, test_iter)

    @classmethod
    def test(cls, x):
        my_config, model = Run.load_model(x)
        model.load_state_dict(torch.load(my_config.save_path))
        _, test_iter = load_data_mnist(my_config.batch_size, resize=my_config.resize)
        test(my_config, model, test_iter)

    @classmethod
    def write_model(cls, x):
        """将模型结构写入tensorboard"""
        writer = SummaryWriter(log_dir='./model_structure/AlexNet/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        my_config, model = Run.load_model(x)
        model.load_state_dict(torch.load(my_config.save_path))
        fake_img = torch.randn(1, 1, 224, 224).to(my_config.device)  # 生成假的图片作为输入
        writer.add_graph(model, fake_img)
        writer.close()


if __name__ == '__main__':
    model_dict = {0: 'AlexNet', 1: 'ResNet'}
    model_name = model_dict[0]
    x = import_module('models.' + model_name)

    Run.seed()

    # Run.write_model(x)
    Run.train(x)
    # Run.test(x)
