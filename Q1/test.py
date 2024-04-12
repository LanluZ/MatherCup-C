import torch

import numpy as np


def test(model_path: str, dataloader):
    def MAELoss(x1: np.ndarray, x2: np.ndarray):
        return np.mean(np.abs(x1 - x2))

    # 载入模型
    model = torch.load(model_path).cuda()
    loss_function = MAELoss
    # 测试模式
    model.eval()

    losses = []
    for i, (inputs, labels) in enumerate(dataloader):
        # 加载数据到GPU
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()

        pred = model(inputs)
        loss = loss_function(pred.cpu().detach().numpy(), labels.cpu().detach().numpy())

        losses.append(loss)

    return losses
