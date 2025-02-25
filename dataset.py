import torchvision
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        自定义数据集，加载图片并返回图像和标签
        :param image_dir: 图片文件夹路径
        :param transform: 对图像的转换（如调整大小、归一化等）
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 获取所有类别文件夹
        self.class_names = os.listdir(image_dir)
        self.class_names.sort()  # 确保类别按字母顺序排序

        # 获取所有图片路径和对应标签
        for label, class_name in enumerate(self.class_names):
            class_folder = os.path.join(image_dir, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    if img_name.endswith('.jpg'):  # 假设图片是 .jpg 格式
                        img_path = os.path.join(class_folder, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        # 标签编码器，将类别名称转换为数值标签
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return image, label


# 定义数据转换
transform_cifar2vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行标准化
])

transform_cifar2res = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行标准化
])

transform_pet2vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

transform_pet2res = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 下载训练数据集
def trainset_path(dataset_name, call_vit=True):
    path_cifar = 'E:\\DATA4DL\\cifar'
    path_pet = 'E:\\DATA4DL\\Oxford-IIIT_Pet\\images'

    if 'cifar100' in dataset_name:
        print('training on cifar100---')
        dataset_call_vit = torchvision.datasets.CIFAR100(root=path_cifar, train=True, download=True, transform=transform_cifar2vit)
        dataset_call_res = torchvision.datasets.CIFAR100(root=path_cifar, train=True, download=True, transform=transform_cifar2res)

        return dataset_call_vit if call_vit else dataset_call_res

    elif 'cifar10' in dataset_name:
        print('training on cifar10---')
        dataset_call_vit = torchvision.datasets.CIFAR10(root=path_cifar, train=True, download=True, transform=transform_cifar2vit)
        dataset_call_res = torchvision.datasets.CIFAR10(root=path_cifar, train=True, download=True, transform=transform_cifar2vit)

        return dataset_call_vit if call_vit else dataset_call_res

    elif 'pet' in dataset_name:
        print('training on Oxford-IIIT_Pet---')
        # 使用自定义数据集类创建训练集和测试集
        train_dir = "E:/DATA4DL/Oxford-IIIT_Pet/split_images/train"  # 训练集路径
        test_dir = "E:/DATA4DL/Oxford-IIIT_Pet/split_images/test"  # 测试集路径

        dataset_call_vit = CustomDataset(image_dir=train_dir, transform=transform_pet2vit)
        dataset_call_res = CustomDataset(image_dir=train_dir, transform=transform_pet2res)

        return dataset_call_vit if call_vit else dataset_call_res

def testset_path(dataset_name, call_vit=True):
    if 'pet' in dataset_name:
        # 使用自定义数据集类创建训练集和测试集
        train_dir = "E:/DATA4DL/Oxford-IIIT_Pet/split_images/train"  # 训练集路径
        test_dir = "E:/DATA4DL/Oxford-IIIT_Pet/split_images/test"  # 测试集路径

        dataset_call_vit = CustomDataset(image_dir=test_dir, transform=transform_pet2vit)
        dataset_call_res = CustomDataset(image_dir=test_dir, transform=transform_pet2res)

        return dataset_call_vit if call_vit else dataset_call_res