import os.path

import pandas as pd
import numpy as np

from datetime import datetime, timedelta


def pretreatment(origin_data, output_path):
    for row in origin_data.iterrows():
        # 日期格式线性化
        date_obj = datetime.strptime(row[1]['日期'], '%Y/%m/%d')
        timestamp = (date_obj - datetime(1970, 1, 1)) // timedelta(hours=1)  # 1970年经过天数
        origin_data.iloc[row[0], 1] = timestamp

        # 补全个位数编号分拣中心
        origin_data.iloc[row[0], 0] = 'SC0' + row[1]['分拣中心'][-1] if len(row[1]['分拣中心']) == 3 else row[1][
            '分拣中心']

    origin_data = origin_data.sort_values(by=['分拣中心', '日期', '小时'], ascending=True)  # 排序和子排序

    # 时间戳
    for i in range(origin_data.shape[0]):
        origin_data.iloc[i, 1] += origin_data.iloc[i, 2]

    # 删除小时列
    origin_data.drop('小时', axis=1, inplace=True)

    # 记录分拣中心分界点
    center_index_data = []
    for i in range(origin_data.shape[0] - 1):
        if origin_data.iloc[i, 0] != origin_data.iloc[i + 1, 0]:
            center_index_data.append((i, origin_data.iloc[i, 0]))  # 记录的最后一小时索引位置与分拣中心名称
    center_index_data.append((origin_data.shape[0] - 1, origin_data.iloc[origin_data.shape[0] - 1, 0]))  # 记录最后一个分拣中心

    origin_data = pd.get_dummies(origin_data, columns=['分拣中心'], prefix='Center')  # 编码哑变量
    origin_data = np.array(origin_data)  # 转化为numpy数组

    # 保存数据
    origin_data = pd.DataFrame(origin_data, index=None)
    center_index_data = pd.DataFrame(center_index_data, index=None)

    origin_data.to_csv(os.path.join(output_path, 'origin_data.csv'), index=False, header=False)
    center_index_data.to_csv(os.path.join(output_path, 'center_index.csv'), index=False, header=False)

    return origin_data, center_index_data
