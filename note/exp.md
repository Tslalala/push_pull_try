**0218 周一**  
**实验0**  
main_ce.py  
~~~python
model = call_model(model_name='vit_base_patch16_224_in21k',   
Prompt_Token_num=10, VPT_type='Deep', frozen_heads=False, classes=35)  
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()
~~~
对pet使用ViT Deep head分类  
结果：  
epoch5 train_acc=83.94% test_acc=81.13%  
epoch10 train_acc=98.13% test_acc=92.12%  
best test_acc=92.83% at epoch15  

**实验1**  
main_ce_proto.py  
~~~python
    model = call_model(model_name='vit_base_patch16_224_in21k', Prompt_Token_num=10, VPT_type='Deep',
                       frozen_heads=False, classes=35)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, device, 20, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_fn=criterion)
~~~
对pet使用ViT Deep proto分类  
结果：  
epoch5 train_acc=84.03% test_acc=89.01%  
epoch10 train_acc=98.31% test_acc=89.11%  
best test_acc=89.69% at epoch14

**实验2**  
main_ce_proto.py  
使用Frozen的ViT，不训练  
Prompt_Token_num=10，test_acc=82.41%,   
Prompt_Token_num= 1, test_acc=84.27%,  
可见使用CrossEntropyLoss能够指引proto方法效果更好，即使得聚类效果更佳

***

**0225 周二**  
**实验3** 
main_pp_.py
预训练ViT在Prompt_Token_num=10时，test_acc=82.41%，  
当我使用下面参数进行PPLoss的指导训练
~~~python
    train_loader = DataLoader(trainset_path('pet'), batch_size=64, shuffle=True)
    test_loader = DataLoader(testset_path('pet'), batch_size=64, shuffle=True)
    model = call_model(model_name='vit_base_patch16_224_in21k', Prompt_Token_num=10, VPT_type='Deep',
                       frozen_heads=True, classes=35)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    train(model, device, 20, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer)
~~~
其中loss_fn = PPLoss(alpha=0.01, beta=0.01, reduction='mean')  
每一个epoch更新一次prototypes  
Train Epoch [1/20]: 100%|██| 70/70 [00:38<00:00, 1.83it/s, accuracy=89.67%]  
Test Epoch [1/20]: 100%|███| 47/47 [00:24<00:00, 1.92it/s, accuracy=88.40%]  
test_acc=88.40%，较实验2的main_ce_proto.py可见PPLoss有效果，  
但是效果不如实验1的main_ce_proto.py，实验1的ce还是现在的PPLoss更猛  

***

**0226 周三**  
**实验4** 
main_pp_.py  
引入了tsne以期待能看到特征空间上更好的聚类效果；引入了数据集pet10，以期待缩短训练时间  
在main_pp_.py上，PPLoss有效果，但是在epoch增加后聚类迅速地变差，  
参数调的好，PPLoss是比不加任何训练的pretrained model的聚类效果要好的，但是比MSELoss尚且比不过  
**需要新的损失函数，可能是分类结果导向的损失函数**

***

**0305 周三**  
**实验5**  
main_ncml.py
Pet还是太大了取Pet里面的10个类组建了Pet10，增加了NCMLoss，  
依旧是ViT-Deep, Prompt_Tokens=10，lr=1e-2，batchsize=80  
Train Epoch [18/20]: 100%|██| 17/17 [00:10<00:00,  1.60it/s, accuracy=98.86%]  
Test Epoch [18/20]: 100%|███| 14/14 [00:07<00:00,  1.97it/s, accuracy=95.34%]  
有一些其他的实验还没有做，让我们补一下ce方法下和pp方法下的这一acc  

在main_ce_proto.py下  
Epoch [17/20]: 100%|███| 17/17 [01:35<00:00,  5.64s/it, accuracy=99.85%, loss=0.0256]
Inference: 100%|███████| 17/17 [00:10<00:00,  1.67it/s]
Test [17/20]: 100%|████| 14/14 [00:07<00:00,  1.99it/s, accuracy=94.77%]

在main_pp_.py下一直跑不起来我也懒得跑了说是  
anyway我们验证了NCMLoss在classes=10时的对NCM分类的有效性是可以与CrossEntropyLoss相当的  
但是ncml可以继续扩展到增量场景下，所以下一步**也许是**在增量场景下研究NCMLoss  
但是老实讲我对NCMLoss还是觉得是否太**简单**，尤其是其更新直接影响到prototypes  
于是我做了这么个实验: 使用NCMLoss但是不更新prototypes，然后我得到了:  
Train Epoch [17/20]: 100%|██| 17/17 [00:10<00:00,  1.62it/s, accuracy=98.71%]  
Test Epoch [17/20]: 100%|███| 14/14 [00:07<00:00,  2.00it/s, accuracy=96.36%]   
就离谱  