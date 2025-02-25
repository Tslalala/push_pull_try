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
使用Frozen的ViT，  
Prompt_Token_num=10，test_acc=82.41%,   
Prompt_Token_num= 1, test_acc=84.27%,  
可见使用CrossEntropyLoss能够指引proto方法效果更好，即使得聚类效果更佳
***
**0225 周二**  
**实验3** 
main_pp.py
预训练ViT在84.27%