import numpy as np

datafile_path = '../data/housing.data'

# 读取数据
data = np.fromfile(datafile_path, sep=" ")

# 特征名称
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 特征数量
feature_num = len(feature_names)

# 数据整形
data = data.reshape([data.shape[0] // feature_num, feature_num])

# 训练比例
ratio = 0.8

offset = int(data.shape[0] * ratio)

min_value = data.min(axis=0)
max_value = data.max(axis=0)

for i in range(data.shape[0]):
    data[i] = (data[i] - min_value) / (max_value - min_value)

training_data = data[:offset]
test_data = data[offset:]
x = training_data[:, :-1]
y = training_data[:, -1:]


class Network:
    def __init__(self, weight_num):
        self.w = np.random.random(13).reshape([13, 1])
        self.b = np.random.random(1).reshape([1, 1])

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return  cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis = 0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z-y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b , eta = 0.01):
        self.w = self.w - gradient_w * eta
        self.b = self.b - gradient_b * eta

    def train(self, x, y, iterations = 10000 , eta = 0.01):
        points = []
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            loss = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(loss)
            if i % 50 == 0:
                print('iter {},  loss {}'.format(i, loss))


net = Network(13)
# 此处可以一次性计算多个样本的预测值和损失函数

net.train(x, y)

test_x = test_data[:, :-1]
test_y = test_data[:, -1:]
test_z = net.forward(test_x)

print("test lost:{}".format(net.loss(test_z, test_y)))