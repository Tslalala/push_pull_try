import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import trainset_path, testset_path
from utils import call_model, classify_with_proto, top_k_accuracy


def train(model, device, num_epochs, train_loader, test_loader, optimizer, loss_fn):
    for epoch in range(num_epochs):
        model.train()  # 确保模型在训练模式
        losses = 0.0
        correct = 0
        total = 0

        # 创建 tqdm 进度条
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]", ncols=120) as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                losses += loss.item()

                # 计算正确预测的数量
                _, predicted = torch.max(outputs, 1)  # 获取预测类别（最大概率的索引）
                correct += (predicted == targets).sum().item()  # 计算正确预测的数量
                total += targets.size(0)  # 累加样本总数

                # 更新进度条中的信息
                accuracy = 100 * correct / total
                pbar.set_postfix(loss=losses / (pbar.n + 1), accuracy=f"{accuracy:.2f}%")  # 显示损失和准确率
                pbar.update(1)  # 更新进度条

        # 推理
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for inputs, targets in tqdm(train_loader, desc="Inference"):
                inputs = inputs.to(device)
                embedding = model.forward_features_(inputs)
                embedding_list.append(embedding.cpu())
                label_list.append(targets.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # NCM, 对class’s features取mean
        class_list = np.unique(label_list)
        feature_proto_list = []
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embeddings = embedding_list[data_index]
            proto = embeddings.mean(0)
            feature_proto_list.append(proto)

        # 测试阶段
        model.eval()

        with tqdm(total=len(test_loader), desc=f"Test Epoch [{epoch + 1}/{num_epochs}]",
                  ncols=120) as pbar_test:
            y_pred, y_true = [], []
            with torch.no_grad():  # 不计算梯度
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model.forward_features_(inputs)
                    loss = loss_fn(outputs, targets)

                    for _, test_output in enumerate(outputs):
                        predict = classify_with_proto(test_output=test_output, prototypes=torch.stack(feature_proto_list))
                        y_pred.append(predict)
                        y_true.append(targets[_].item())

                    # 更新测试进度条信息
                    test_accuracy = top_k_accuracy(y_pred=y_pred, y_true=y_true, k=1)
                    pbar_test.set_postfix(accuracy=f"{test_accuracy:.2f}%")
                    pbar_test.update(1)

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = 'pet10'
    train_loader = DataLoader(trainset_path(data_name), batch_size=80, shuffle=True)
    test_loader = DataLoader(testset_path(data_name), batch_size=64, shuffle=True)
    model = call_model(model_name='vit_base_patch16_224_in21k', Prompt_Token_num=10, VPT_type='Deep',
                       frozen_heads=False, classes=35)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, device, 20, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_fn=criterion)


if __name__ == '__main__':
    main()