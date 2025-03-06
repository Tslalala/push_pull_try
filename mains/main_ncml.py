import torch
import numpy as np
from pyexpat import features
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset import trainset_path, testset_path
from utils import call_model, get_protos, test_accuracy, get_protos_with_tqdm, tsne_classes
from losses import PPLoss, NCMLoss

'''test'''
def train(model, device, num_epochs, train_loader, test_loader, optimizer):

    # 实例化PPLoss，第一次推理得到prototype
    loss_fn = NCMLoss()
    feature_proto_list = torch.stack(get_protos_with_tqdm(data_loader=train_loader, device=device, model=model)).to(device)
    # test_accuracy(model=model, data_loader=train_loader, prototypes=feature_proto_list, epoch=0,
    #               num_epochs=1, device=device, words='Train')
    # test_accuracy(model=model, data_loader=test_loader, prototypes=feature_proto_list, epoch=0,
    #               num_epochs=1, device=device, words='Test')

    for epoch in range(num_epochs):
        model.train()  # 确保模型在训练模式
        losses = 0.0

        # 创建 tqdm 进度条
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]", ncols=120) as pbar:
            for batch_counter, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # 准备tsne数据
                target_long = targets.long().detach().squeeze()
                if batch_counter == 0:
                    feature_bank, target_bank = outputs.detach().cpu(), target_long.detach().cpu()
                else:
                    feature_bank = torch.cat((feature_bank, outputs.detach().cpu()), dim=0)
                    target_bank = torch.cat((target_bank, target_long.detach().cpu()), dim=0)

                loss = loss_fn.forward(features=outputs, labels=targets, prototypes=feature_proto_list)
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # 更新进度条中的信息
                pbar.set_postfix(loss=losses / (pbar.n + 1))  # 显示损失和准确率
                pbar.update(1)  # 更新进度条

                # 不想跑完train_loader
                # if batch_counter == 15:
                #     break

                # 在训练时更新prototypes, 通常没太多用
                # if batch_counter % 5 == 1:
                #     feature_proto_list = torch.stack(get_protos(data_loader=train_loader, device=device, model=model)).to(device)

        # 推理得到prototypes
        # feature_proto_list = torch.stack(get_protos_with_tqdm(data_loader=train_loader, device=device, model=model)).to(device)
        test_accuracy(model=model, data_loader=train_loader, prototypes=feature_proto_list, epoch=epoch,
                      num_epochs=num_epochs, device=device, words='Train')
        test_accuracy(model=model, data_loader=test_loader, prototypes=feature_proto_list, epoch=epoch,
                      num_epochs=num_epochs, device=device, words='Test')

        # 绘tsne图
        tsne_classes(feature_bank, target_bank)

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_name = 'pet10'
    train_loader = DataLoader(trainset_path(dataset_name=data_name), batch_size=80, shuffle=True)
    test_loader = DataLoader(testset_path(dataset_name=data_name), batch_size=64, shuffle=True)
    model = call_model(model_name='vit_base_patch16_224_in21k', Prompt_Token_num=10, VPT_type='Deep',
                       frozen_heads=True, classes=35)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train(model, device, 20, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer)


if __name__ == '__main__':
    main()