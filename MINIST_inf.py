import torchvision

# 统计minist训练集和测试集各样本数量
minist_train = torchvision.datasets.MNIST(root='.\data', train=True, download=False)
minist_test = torchvision.datasets.MNIST(root='.\data', train=False, download=False)
train_num = {i: 0 for i in range(10)}
test_num = {i: 0 for i in range(10)}

for sample in minist_train:
    train_num[sample[1]] += 1

for sample in minist_test:
    test_num[sample[1]] += 1

print('train\n')
for num in train_num.items():
    print(num[1])

print('\ntest\n')
for num in test_num.items():
    print(num[1])
