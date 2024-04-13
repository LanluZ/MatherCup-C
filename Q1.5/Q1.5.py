import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

from model import *
from train import *
from test import *
from predict import *

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
    seq_length = 57  # 序列大小
    predict_seq_time = 57 * 30  # 预测长度

    epochs = 20  # 训练轮次
    batch_size = 40  # 批次大小
    learn_rate = 0.001  # 学习率

    create_model_mode = False  # 创建新模型
    train_model_mode = False  # 训练模型
    test_model_mode = False  # 测试模型
    convert_model_mode = False  # 转换模型
    predict_model_mode = True  # 预测模型

    # 读取数据
    origin_data = pd.read_csv(os.path.join(data_path, '附件1.csv'), encoding='GBK')

    # 数据预处理
    for row in origin_data.iterrows():
        # 日期格式线性化
        date_obj = datetime.strptime(row[1]['日期'], '%Y/%m/%d')
        timestamp = (date_obj - datetime(1970, 1, 1)) // timedelta(days=1)  # 1970年经过天数
        origin_data.iloc[row[0], 1] = timestamp

        # 补全个位数编号分拣中心
        origin_data.iloc[row[0], 0] = 'SC0' + row[1]['分拣中心'][-1] if len(row[1]['分拣中心']) == 3 else row[1][
            '分拣中心']

    origin_data = origin_data.sort_values(by=['日期', '分拣中心', '小时'], ascending=True)  # 排序和子排序
    origin_data = pd.get_dummies(origin_data, columns=['分拣中心'], prefix='Center')  # 编码哑变量
    origin_data = np.array(origin_data)  # 转化为numpy数组

    # 按时间序列升维
    data_x, data_y = [], []
    origin_data_x = scaler_x.fit_transform(origin_data[:, :])  # 归一化原始数据x
    origin_data_y = scaler_y.fit_transform(np.expand_dims(origin_data[:, 1], axis=1))  # 归一化原始数据y
    for i in range(origin_data.shape[0] - seq_length):
        data_x.append(origin_data_x[i:i + seq_length])  # 现在
        data_y.append(origin_data_y[i + 1:i + seq_length + 1])  # 未来
    data_x = np.array(data_x)  # 三维
    data_y = np.array(data_y)  # 三维

    # 格式转换
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)

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
        predict_y = predict(model_pkl_path, origin_data, seq_length, predict_seq_time, scaler_x, scaler_y)
        predict_y[:, 1] = predict_y[:, 1].astype(np.integer)  # 格式整理
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
