import numpy as np
import os
import glob
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.initializers import truncated_normal
from keras.layers import Flatten, Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam
 
# 训练用图片存放路径
p_path = '/home/beckhans/Projects/FaceCNN/data'
 
# 将所有的图片resize成128*128
w = 128
h = 128
c = 3
 
 
def read_img(path):
    # 读取当前文件夹下面的所有子文件夹
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    # 图片数据集
    imgs = []
    # 标签数据集
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % im)
            # 读取照片
            img = cv2.imread(im)
            # 将照片resize128*128*3
            img = cv2.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

def cnnlayer():
    # 第一个卷积层（128——>64)
    model = Sequential()
    conv1 = Conv2D(
        input_shape=[128, 128, 3],
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None))
    pool1 = MaxPooling2D(pool_size=(2, 2),
                         strides=2)
    model.add(conv1)
    model.add(pool1)
    # 第二个卷积层(64->32)
    conv2 = Conv2D(filters=64,
                   kernel_size=(5, 5),
                   padding="same",
                   activation="relu",
                   kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None))
    pool2 = MaxPooling2D(pool_size=[2, 2],
                         strides=2)
    model.add(conv2)
    model.add(pool2)
 
    # 第三个卷积层(32->16)
    conv3 = Conv2D(filters=128,
                   kernel_size=[3, 3],
                   padding="same",
                   activation="relu",
                   kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None))
    pool3 = MaxPooling2D(pool_size=[2, 2],
                         strides=2)
    model.add(conv3)
    model.add(pool3)
 
    # 第四个卷积层(16->8)
    conv4 = Conv2D(filters=128,
                   kernel_size=[3, 3],
                   padding="same",
                   activation="relu",
                   kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None))
    pool4 = MaxPooling2D(pool_size=[2, 2],
                         strides=2)
    model.add(conv4)
    model.add(pool4)
    model.add(Flatten())
    model.add(Dropout(0.5))
    # 全连接层
    dense1 = Dense(units=1024,
                   activation="relu",
                   kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None),
                   kernel_regularizer=regularizers.l2(0.003))
    model.add(Dropout(0.5))
    dense2 = Dense(units=512,
                   activation="relu",
                   kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None),
                   kernel_regularizer=regularizers.l2(0.003))
    model.add(Dropout(0.5))
    dense3 = Dense(units=5,
                   activation='softmax',
                   kernel_initializer=truncated_normal(mean=0.0, stddev=0.01, seed=None),
                   kernel_regularizer=regularizers.l2(0.003))
    model.add(dense1)
    model.add(dense2)
    model.add(dense3)
    return model

if __name__ == '__main__':
    data, label = read_img(p_path)
 
    num_example = data.shape[0]
    arr = np.arange(num_example)
    
    # 打乱顺序
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
 
    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    y_train = to_categorical(y_train, num_classes=5)
    x_val = data[s:]
    y_val = label[s:]
    y_val = to_categorical(y_val, num_classes=5)
 
    # 使用Adam优化器
    model = cnnlayer()
    
    # 设定学习率0.01
    adam = Adam(lr=0.001)
    
    # 使用分类交叉熵作为损失函数
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
 
    # 训练模型
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_val, y_val, batch_size=32)
 
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    # 模型保存
    model.save("CNN.model")