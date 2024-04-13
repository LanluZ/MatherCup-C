import torch
import copy

import numpy as np
import pandas as pd


def predict(model_path, origin_x, predict_seq_time, scaler_x, scaler_y):
    model = torch.load(model_path).cuda()
    model.eval()  # 测试模式
    # 循环预测
    predict_y = []
    for i in range(predict_seq_time):
        x = scaler_x.transform(origin_x)  # 归一化
        x = np.expand_dims(x, axis=0)  # 升维
        y = model(torch.tensor(x).float().cuda())  # 预测
        y = y.cpu().detach().numpy().squeeze(0)  # 降维
        result = scaler_y.inverse_transform(y)  # 反归一化
        result = result.squeeze(1)  # 降维

        # 新行准备
        new_line_x = copy.copy(origin_x[0])
        new_line_x[0] += origin_x.shape[0]  # 日期更改
        new_line_x[1] = result[-1]  # 货值更改

        # 旧行删除
        origin_x = np.delete(origin_x, 0, axis=0)

        # 新行拼接
        origin_x = np.vstack((origin_x, new_line_x))

        # 结果记录
        predict_y.append(new_line_x[0:2])

    # 结果保存
    predict_y = np.array(predict_y)
    return predict_y
