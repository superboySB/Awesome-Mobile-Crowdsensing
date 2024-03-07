pos_features = 2
labels_number = 100

import numpy as np
import torch
import torch.nn as nn


# 定义模型
class DistancePredictor(nn.Module):
    def __init__(self):
        super(DistancePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(23, 64),  # 输入是两个点的坐标，总共4个值
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出是一个值，即两点之间的距离
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.fc(x)


def train():
    import torch.optim as optim

    # 创建数据集
    def create_dataset(num_samples):
        # points = np.random.rand(num_samples, 2, 2)  # 生成随机点对
        # distances = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)  # 计算欧氏距离
        # return torch.tensor(points, dtype=torch.float32), torch.tensor(distances, dtype=torch.float32)

        # agent_position = np.tile(np.random.rand(pos_features), reps=num_samples).reshape(num_samples, pos_features)
        agent_position = np.random.rand(num_samples, pos_features)
        pois_position = np.random.rand(num_samples, pos_features)
        aois = np.random.randint(0, 20, (num_samples, 1))
        noise = np.random.normal(0, 0.1, (num_samples, 18))
        distances = np.linalg.norm(pois_position - agent_position, axis=1)
        labels = distances * aois[..., 0] / 10.0
        # add noise (which prevent the test loss from reaching low value)
        # distances += np.random.normal(0, 0.1, num_samples)
        inputs = np.concatenate([agent_position, noise, pois_position, aois], axis=1)
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    # 参数设置
    num_samples = 2000
    epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # 数据准备
    points, distances = create_dataset(num_samples)
    dataset = torch.utils.data.TensorDataset(points, distances)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数和优化器
    model = DistancePredictor().to('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epochs):
        for batch_points, batch_distances in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_points.cuda())
            loss = criterion(outputs.squeeze(), batch_distances.cuda())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # 测试模型
        evaluate(create_dataset, criterion, model)
    evaluate(create_dataset, criterion, model, True)


def evaluate(create_dataset, criterion, model, verbose=False):
    with torch.no_grad():
        test_points, labels = create_dataset(10)
        predictions = model(test_points.cuda())
        if verbose:
            # print prediction and label one to one
            for i in range(10):
                print(f'Prediction: {predictions[i].item()}, Label: {labels[i].item()}')
        # print test loss
        test_loss = criterion(predictions.squeeze(), labels.cuda())
        print(f'Test loss: {test_loss.item()}')


if __name__ == '__main__':
    train()

    # from tqdm import tqdm
    # # fix random seed for np and torch
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # agent_position = np.tile(np.random.rand(pos_features), reps=labels_number).reshape(labels_number, pos_features)
    # print(agent_position[0])
    # pois_position = np.random.rand(labels_number, pos_features)
    # distances = np.linalg.norm(pois_position - agent_position, axis=1)
    # predictor = nn.Sequential(
    #     nn.Linear(pos_features * 2, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 1)
    # ).to('cuda')
    # predictor.train()
    # loss_func = nn.MSELoss()
    # optimizer = torch.optim.Adam(predictor.parameters(), lr=0.0001)
    # progress = tqdm(range(1000))
    # agent_position_tensor = torch.from_numpy(agent_position).to(torch.float32).cuda()
    # pois_position_tensor = torch.from_numpy(pois_position).to(torch.float32).cuda()
    #
    # for epoch in progress:
    #     labels = torch.from_numpy(distances).to(torch.float32).cuda()
    #     labels = labels.detach()
    #     labels.requires_grad = False
    #     inputs = torch.cat([agent_position_tensor, pois_position_tensor], dim=1)
    #     # use minibatch
    #     outputs = predictor(inputs)
    #     loss = loss_func(outputs, labels)
    #     # progress.set_postfix({'loss': loss.item()})
    #     print(f"loss {loss.item()}")
    #     # loss = torch.mean(torch.abs(outputs - labels))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # predictor.eval()
    # test_pois_position = np.random.rand(labels_number, pos_features)
    # test_pois_position_tensor = torch.from_numpy(test_pois_position).to(torch.float32).cuda()
    # test_distances = np.linalg.norm(test_pois_position - agent_position, axis=1)
    # test_labels = torch.from_numpy(test_distances).to(torch.float32).cuda()
    # test_labels = test_labels.detach()
    # test_labels.requires_grad = False
    # test_inputs = torch.cat([agent_position_tensor, test_pois_position_tensor], dim=1)
    # test_outputs = predictor(test_inputs)
    # test_loss = loss_func(test_outputs, test_labels)
    # print(f"test loss {test_loss.item()}")
    # print(test_labels[:10])
    # print(test_outputs[:10])
