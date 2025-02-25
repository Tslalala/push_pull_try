import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from convs.vpt import build_promptmodel
from convs.resnet_big import SupConResNet


def call_model(model_name='vit_base_patch16_224_in21k', Prompt_Token_num=5, VPT_type='Shallow',
               frozen_heads=True, classes=10):
    if 'vit' in model_name:
        print(f'Frozen heads {frozen_heads}')

        model = build_promptmodel(modelname=model_name, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type,
                                  frozen_heads=frozen_heads, new_classes=classes)

        # Freeze the parameters for ViT.
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        return model

    elif 'res' in model_name:
        print('Using ResNet model')
        model = SupConResNet(name='resnet50', head='mlp', feat_dim=128)

        # Freeze the parameters for ResNet50.
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        return model

    else:
        raise NotImplementedError

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def classify_with_proto(test_output, prototypes, top_num=2):
    """
    基于原型（L2距离）的分类函数
    Args:
        test_output: 测试样本特征 [768]
        prototypes: 所有类别原型 [35, 768]
        top_num: 返回前top_num个预测结果
    Returns:
        predict: 预测的类别索引列表（长度=top_num）
    """
    # 确保数据在CPU（若prototypes在GPU）
    test_output = test_output.cpu()
    prototypes = prototypes.cpu()

    # 向量化计算所有类别的L2距离（形状 [35]）
    distances = torch.sum((test_output - prototypes) ** 2, dim=1)  # [35]

    # 计算相似度（取距离倒数，避免除零）
    epsilon = 1e-8  # 防止距离为0导致无穷大
    similarities = 1 / (distances + epsilon)  # [35]

    # 直接取Top-K类别索引
    top_sim, top_indices = torch.topk(similarities, k=top_num, largest=True, sorted=True)
    predict = top_indices.tolist()

    return predict[:top_num]  # 确保返回长度一致

def top_k_accuracy(y_pred, y_true, k):
    correct_count = 0
    total_count = len(y_true)

    for i in range(total_count):
        # 检查真实标签是否在预测的前 k 个类别中
        if y_true[i] in y_pred[i][:k]:
            correct_count += 1

    # 计算 top-k 准确率
    accuracy = correct_count / total_count * 100
    return accuracy

def get_protos_with_tqdm(data_loader, device, model):
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Inference"):
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

    return feature_proto_list

def get_protos(data_loader, device, model):
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for inputs, targets in data_loader:
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

    return feature_proto_list

def test_accuracy(model, data_loader, prototypes, epoch, num_epochs, device, words='Test'):
    model.eval()

    with tqdm(total=len(data_loader), desc=f"{words} Epoch [{epoch + 1}/{num_epochs}]",
              ncols=120) as pbar_test:
        y_pred, y_true = [], []
        with torch.no_grad():  # 不计算梯度
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                for _, test_output in enumerate(outputs):
                    predict = classify_with_proto(test_output=test_output, prototypes=prototypes)
                    y_pred.append(predict)
                    y_true.append(targets[_].item())

                # 更新测试进度条信息
                test_accuracy = top_k_accuracy(y_pred=y_pred, y_true=y_true, k=1)
                pbar_test.set_postfix(accuracy=f"{test_accuracy:.2f}%")
                pbar_test.update(1)

    return test_accuracy