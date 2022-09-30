import os
import struct

import numpy as np

# 当前文件目录
current_path = os.path.dirname(__file__)

# 数据集目录
dataset_path = current_path + "/../data/handwritten_digit/"

# 训练图片文件名
train_images_path = "train-images-idx3-ubyte"
# 训练标签文件名
train_labels_path = "train-labels-idx1-ubyte"

# 文件的比特形式
train_images_bytes = open(dataset_path + train_images_path, "rb").read()

# 偏移量
offset = 0

# 文件头格式
fmt_header = ">iiii"

# 解析文件
magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, train_images_bytes, offset)

print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

# 图片尺寸
image_size = num_rows * num_cols

# 偏移文件头
offset += struct.calcsize(fmt_header)

# 图片文件格式
fmt_image = ">" + str(image_size) + "B"

# 训练数据集，矩阵大小：（图片数量，图片行数，图片列数）
train_data = np.empty((num_images, num_rows, num_cols))

for i in range(num_images):
    # 逐个读取图片
    train_data[i] = np.array(struct.unpack_from(fmt_image, train_images_bytes, offset)).reshape((num_rows, num_cols))
    # 计算偏移
    offset += struct.calcsize(fmt_image)

# 读取训练标签数据集
fmt_header = ">ii"
offset = 0
train_labels_bytes = open(dataset_path + train_labels_path, "rb").read()

# magic_number 魔数，文件头
# num_labels，标签数量
magic_number, num_labels = struct.unpack_from(fmt_header, train_labels_bytes, offset)

train_labels = np.empty(num_labels, dtype="int")
offset += struct.calcsize(fmt_header)
fmt_label = ">" + str(1) + "B"
for i in range(num_labels):
    train_labels[i] = np.array(struct.unpack_from(fmt_label, train_labels_bytes, offset), dtype="int")
    offset += struct.calcsize(fmt_label)


# 训练集和标签已经处理好了
# 训练集：train_data
# 标签：train_label

# 如果我们使用线性模型进行训练
# 数学模型是咋样的？

class Network:
    def __init__(self, weight_num, laber_num):
        self.w = np.random.random(weight_num * laber_num).reshape([weight_num, laber_num])
        self.b = np.random.random(laber_num).reshape([1, laber_num])

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        np.exp(z)
        z = z / np.sum(x, axis=1)
        return z

    def loss(self, z, y):
        softmax_z = (-1) * np.log(z) * z * y
        cost = np.mean(softmax_z)
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x 
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - gradient_w * eta
        self.b = self.b - gradient_b * eta

    def train(self, x, y, iterations=10000, eta=0.01):
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
