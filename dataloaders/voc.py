# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(BaseDataSet):

    def __init__(self, **kwargs):
        # self.num_classes = 3 设置这个数据集有21个类别（包括背景）。
        # self.palette = palette.get_voc_palette(self.num_classes) 获取一个用于颜色映射的调色板，这在将分割结果可视化时很有用。
        self.num_classes = 3
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        # 指定数据集的根目录、图像目录和标签目录。
        # 根据提供的数据集分割（如'train', 'val'）读取相应的图像列表。
        self.root = os.path.join(self.root, 'BubbleDataSet')
        # self.image_dir 设置为存储JPEG图像的目录的路径。
        # self.label_dir 设置为存储分割标签的目录的路径。
        # 这两个目录分别存放着数据集中的图像和它们对应的标注信息。
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        # 这两行代码读取一个文本文件，该文件列出了属于特定分割（如'train', 'val'等）的图像的文件名。
        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        # open(file_list, "r") 打开文件进行读取。tuple(...) 将文件中的每一行转换为一个元组的元素。
        # 列表推导式 [line.rstrip() for line in ...] 用于遍历文件的每一行，line.rstrip() 删除每行末尾的空格和换行符。
        # 最终，self.files 被赋值为一个列表，包含了该数据集分割中所有图像的文件名。
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        # 根据索引加载指定的图像和对应的标签文件。
        # 图像和标签都被转换为NumPy数组格式。

        # 从 self.files 列表中获取指定索引 (index) 的图像ID。这个列表在 _set_files 方法中被填充，包含了数据集中所有图像的文件名。
        image_id = self.files[index]
        # image_path 是指定图像文件的完整路径。它是通过将图像目录的路径 (self.image_dir) 与图像的文件名（并附上适当的文件扩展名）拼接起来构成的。
        # label_path 是对应标签文件的完整路径。同样，它是通过将标签目录的路径 (self.label_dir) 与标签的文件名拼接而成。
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        # 使用PIL库 (Image.open) 加载图像文件，并将其转换为NumPy数组，数据类型设置为 float32。这种类型转换常用于确保图像数据以适当的格式输入到深度学习模型中。
        # 同样的操作也用于加载标签文件，但数据类型设置为 int32，因为标签通常是整数形式的类别ID。
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        # 这行代码再次从 self.files 中获取图像ID，并通过分割字符串的方式提取图像的唯一标识符。
        # 这个标识符通常用于在训练过程中识别或引用特定的图像。
        image_id = self.files[index].split("/")[-1].split(".")[0]
        # 方法返回加载的图像和标签（作为NumPy数组），以及图像的唯一标识符。
        # 这些返回值将用于训练或评估模型时提供必要的数据和元数据信息。
        return image, label, image_id
        # 输出image和label的数组，和数字形式的图片名字

class VOCAugDataset(BaseDataSet):
    # VOCAugDataset 类继承自 BaseDataSet，用于处理包含额外注释（增强数据集）的VOC数据集。

    def __init__(self, **kwargs):
        # 类似于 VOCDataset，设定类别数和调色板。
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        # 指定数据集的根目录。
        # 读取图像和标签的文件列表，这些文件可能包含来自额外数据源的标注。
        self.root = os.path.join(self.root, 'BubbleDataSet')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))
    
    def _load_data(self, index):
        # 类似于 VOCDataset，加载图像和对应的标签。
        image_path = os.path.join(self.root, self.files[index][1:])
        label_path = os.path.join(self.root, self.labels[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class VOC(BaseDataLoader):
    # VOC 类继承自 BaseDataLoader，是用于加载Pascal VOC数据集的数据加载器。
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
    # 接收一系列参数，如数据目录、批次大小、图像的裁剪和缩放设置等。
    # 设置图像处理时使用的均值和标准差。
    # 根据分割名称（如 'train', 'val', 'train_aug'）决定使用 VOCDataset 还是 VOCAugDataset。
    # 调用基类 BaseDataLoader 的初始化方法来设置数据集、批量大小、是否打乱数据等。

        # 这两行代码设置了用于图像标准化的均值（MEAN）和标准差（STD）。这些值用于图像预处理，通常是针对特定数据集进行调整的。

        # 这个列表表示图像在RGB三个通道上的全局平均像素值。
        # 在预处理时，这个均值会从每个通道的每个像素值中减去，使得数据中心化（Centering），即数据的平均值为0。
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        # 这个列表表示图像在RGB三个通道上的标准差。
        # 在预处理时，每个通道的像素值减去均值后，再除以这个标准差，使得数据的分布在每个通道上具有单位标准差（Scaling）。
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        # 这里构建了一个字典 kwargs，包含了数据加载和预处理过程中需要的参数。
        kwargs = {
            'root': data_dir,
            # 数据集的根目录路径。这是存储数据集文件（如图像和标签）的位置。
            'split': split,
            # 指定数据集的哪一部分被使用，例如 'train'、'val'（验证）等。这通常用于指定是训练集、验证集还是测试集。
            'mean': self.MEAN,
            # 图像预处理时使用的通道均值，用于数据标准化。这个均值是从数据集中计算得出的。
            'std': self.STD,
            # 图像预处理时使用的通道标准差，同样用于数据标准化。
            'augment': augment,
            # 布尔值，指示是否应用数据增强。数据增强包括各种技术，如随机裁剪、旋转等，用于增加数据的多样性和改善模型的泛化能力。
            'crop_size': crop_size,
            # 在数据增强或预处理时应用的裁剪大小。这通常是一个元组，指定裁剪后图像的高度和宽度。
            'base_size': base_size,
            # 基础大小，用于在裁剪之前缩放图像。这有助于在裁剪操作中保持一定的尺寸比例。
            'scale': scale,
            # 布尔值，指示是否在预处理时对图像进行缩放。
            'flip': flip,
            # 布尔值，指示是否在数据增强时应用随机水平翻转。
            'blur': blur,
            # 布尔值，指示是否应用模糊效果作为数据增强的一部分。
            'rotate': rotate,
            # 布尔值，指示是否在数据增强时应用随机旋转。
            'return_id': return_id,
            # 布尔值，指示是否在加载数据时返回每个样本的唯一标识符或名称。
            'val': val
            # 布尔值，指示当前是否处理验证数据集。这通常影响数据加载和预处理的方式。
        }

        # 根据 split 参数的值选择不同的数据集类。split 指定了数据集的哪一部分将被使用，例如 "train"、"val" 或它们的增强版本 "train_aug"、"val_aug" 等。
        # 如果 split 是增强版本，使用 VOCAugDataset 类；如果是标准版本，使用 VOCDataset 类。
        # 如果 split 不是这些预定义的值之一，抛出一个值错误。
        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = VOCAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = VOCDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")

        # 这行代码调用了基类 BaseDataLoader 的构造函数，将 self.dataset 和其他参数传递给它。
        # 这样，VOC 类的实例会拥有 BaseDataLoader 的所有功能，比如能够进行批量加载、数据打乱、多线程处理等。
        super(VOC, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

    # data_dir
    # 含义：数据集的根目录路径。这是存储数据集文件（如图像和标签）的位置。
    # batch_size
    # 含义：每次迭代训练时加载的样本数量。批量大小会影响训练的内存需求和优化过程。
    # split
    # 含义：指定数据集的哪一部分被使用，例如 'train'、'val'（验证）等。这通常用于指定是训练集、验证集还是测试集。
    # crop_size
    # 含义：（可选）在数据增强或预处理时应用的裁剪大小。这通常是一个元组，指定裁剪后图像的高度和宽度。
    # base_size
    # 含义：（可选）基础大小，用于在裁剪之前缩放图像。这有助于在裁剪操作中保持一定的尺寸比例。
    # scale
    # 含义：布尔值，指示是否在预处理时对图像进行缩放。
    # num_workers
    # 含义：用于数据加载的线程数。更多的线程可以加快数据加载的速度，但也会增加内存消耗。
    # val
    # 含义：布尔值，指示当前是否处理验证数据集。这通常影响数据加载和预处理的方式。
    # shuffle
    # 含义：布尔值，指示是否在每个训练周期开始时打乱数据。
    # flip
    # 含义：布尔值，指示是否在数据增强时应用随机水平翻转。
    # rotate
    # 含义：布尔值，指示是否在数据增强时应用随机旋转。
    # blur
    # 含义：布尔值，指示是否应用模糊效果作为数据增强的一部分。
    # augment
    # 含义：布尔值，指示是否应用数据增强。数据增强包括各种技术，如随机裁剪、旋转等，用于增加数据的多样性和改善模型的泛化能力。
    # val_split
    # 含义：（可选）指定从训练数据中分割出一部分作为验证集。
    # return_id
    # 含义：布尔值，指示是否在加载数据时返回每个样本的唯一标识符或名称。
