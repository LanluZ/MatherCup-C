import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from model import *
from train import *
from test import *
from predict import *
from pretreatment import *

# 环境获取
project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, 'data')
output_path = os.path.join(project_path, 'output')
model_pkl_path = os.path.join(output_path, 'model.pkl')
model_onnx_path = os.path.join(output_path, 'model.onnx')

# 归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


def main():
    input_size = 59  # 输入层维度
    hidden_size = 512  # 隐藏层维度
    num_layers = 2  # 堆叠层数
    output_size = 1  # 输出层维度
    seq_length = 24  # 序列大小
    predict_seq_time = 24 * 30  # 预测长度

    epochs = 20  # 训练轮次
    batch_size = 40  # 批次大小
    learn_rate = 0.001  # 学习率

    pretreatment_data_mode = True  # 是否预处理原始数据
    create_model_mode = False  # 创建新模型
    train_model_mode = False  # 训练模型
    test_model_mode = False  # 测试模型
    convert_model_mode = False  # 转换模型
    predict_model_mode = False  # 预测模型

    # 数据预处理
    if pretreatment_data_mode:
        origin_data = pd.read_csv(os.path.join(data_path, '附件2.csv'), encoding='GBK')
        pretreatment(origin_data, output_path)

    # 读取数据
    origin_data = pd.read_csv(os.path.join(output_path, 'origin_data.csv'), header=None)
    center_index = pd.read_csv(os.path.join(output_path, 'center_index.csv'), header=None)

    # 格式转换
    origin_data = np.array(origin_data)
    center_index = np.array(center_index)

    # 按时间序列升维
    data_x, data_y = [], []
    origin_data_x = scaler_x.fit_transform(origin_data[:, :])  # 归一化原始数据x
    origin_data_y = scaler_y.fit_transform(np.expand_dims(origin_data[:, 1], axis=1))  # 归一化原始数据y
    for i in range(origin_data.shape[0] - seq_length - 1):
        if i + seq_length not in center_index:
            data_x.append(origin_data_x[i:i + seq_length])  # 现在
            data_y.append(origin_data_y[i + 1:i + seq_length + 1])  # 未来
    data_x = np.array(data_x)  # 三维
    data_y = np.array(data_y)  # 三维

    # 格式转换
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)

    print('//完成数据加载//')

    # 创建数据集
    train_data_x = data_x[int(data_x.shape[1] * 0.7):]  # 各种变量
    train_data_y = data_y[int(data_y.shape[1] * 0.7):]  # 货量

    test_data_x = data_x[:int(data_y.shape[1] * 0.7)]  # 各种变量
    test_data_y = data_x[:int(data_y.shape[1] * 0.7)]  # 货量

    train_dataset = LSTMDataset(train_data_x, train_data_y)
    test_dataset = LSTMDataset(test_data_x, test_data_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 创建模型
    if create_model_mode:
        model = LSTM(input_size, hidden_size, num_layers, output_size)
        torch.save(model, model_pkl_path)

    # 训练模型
    if train_model_mode:
        train_loss = train(model_pkl_path, train_dataloader, epochs, learn_rate)
        train_loss = pd.DataFrame(train_loss, index=None)
        train_loss.to_csv(os.path.join(output_path, 'train_loss.csv'), header=False, index=False)

    # 测试模型
    if test_model_mode:
        test_loss = test(model_pkl_path, test_dataloader)
        test_loss = pd.DataFrame(test_loss, index=None)
        test_loss.to_csv(os.path.join(output_path, 'test_loss.csv'), header=False, index=False)

    # 转换模型
    if convert_model_mode:
        model = torch.load(model_pkl_path).cpu()
        model.eval()  # 测试模式
        inputs = torch.randn(1, seq_length, input_size)
        torch.onnx.export(model, inputs, model_onnx_path)

    # 预测模型
    if predict_model_mode:
        origin_x = origin_data[-seq_length:]
        predict_y = predict(model_pkl_path, origin_x, predict_seq_time, scaler_x, scaler_y)
        predict_y[:, 2] = predict_y[:, 2].astype(np.integer)  # 格式整理
        predict_y[:, 0] = predict_y[:, 0] * timedelta(days=1) + datetime(1970, 1, 1)  # 时间戳反算
        predict_y = pd.DataFrame(predict_y, index=None)
        predict_y.to_csv(os.path.join(output_path, 'prediction.csv'), header=False, index=False)


class LSTMDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x  # 数据列x
        self.data_y = data_y  # 数据列y

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return self.data_x.shape[0]


if __name__ == '__main__':
    main()
