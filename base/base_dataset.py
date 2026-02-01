import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import os

class BaseDataSet(Dataset):
    # 这行定义了一个名为 BaseDataSet 的类，它继承自 PyTorch 的 Dataset 类。
    # 这意味着 BaseDataSet 类将实现 Dataset 所需要的方法（例如 __len__ 和 __getitem__），并可以添加额外的功能。
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
    # 这是 BaseDataSet 类的构造函数，用于初始化数据集的实例。
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
        # 如果启用了数据增强 (self.augment 为真)，则将相应的参数赋值给类的实例变量。
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        # 这里transforms.ToTensor()是一个torchvision库中的变换操作，它也可以将numpy数组或者PIL图片转换成tensor。
        # 与torch.tensor()不同的是，transforms.ToTensor()通常在定义数据预处理流水线时使用，它不仅转换数据类型，还会自动将数据的值范围从[0, 255]缩放到[0.0, 1.0]。
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)
        # 设置 OpenCV 在进行图像操作时使用的线程数。cv2.setNumThreads(0) 使 OpenCV 自动选择线程数，通常能提高处理效率。

    def _set_files(self):
        raise NotImplementedError
    # raise NotImplementedError 是一种在 Python 中常用的编程习惯，它用于指示某个方法或功能还没有被实现，通常出现在基类（或抽象类）中的方法里。
    # 在面向对象编程中，这种做法有助于提供一个统一的接口，同时让子类去实现具体的细节。
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # interpolation=cv2.INTER_LINEAR 参数指定了缩放时使用的插值方法，这里使用的是线性插值，适合于缩放时保持图像平滑。
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            # 首先，Image.fromarray(label) 将 label 数组转换为 PIL 图像对象。
            # 接着使用 .resize((w, h), resample=Image.NEAREST) 方法调整图像大小，使其与之前调整过的图像大小相匹配。
            # Image.NEAREST 作为重采样方法，这是最近邻插值，适合于处理标签图像，因为它不会引入新的颜色标签（类别），保持了标签的原始性。
            label = np.asarray(label, dtype=np.int32)
            # 这行代码将调整大小后的 PIL 图像对象转换回 NumPy 数组格式。
            # np.asarray(label, dtype=np.int32) 将 PIL 图像对象转换为一个 NumPy 数组，dtype=np.int32 指定数组的数据类型为 32 位整数。

            # 中心裁切
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
            
            # 保存增强后的图像和标签
            cv2.imwrite(os.path.join('/home/wch/Deeplearning/CV/deeplab/check', 'enhanced_image.jpg'), image)  # 保存图像
            label_pil = Image.fromarray(label.astype(np.uint8))  # 转换为PIL图像，确保类型正确
            label_pil.save(os.path.join('/home/wch/Deeplearning/CV/deeplab/check', 'enhanced_label.png'))  # 保存标签图像

        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
                # 如果启用了缩放（self.scale为真），则随机选择一个长度在base_size的一半到两倍之间的数作为图像的“长边”尺寸。
                # 如果没有启用缩放，则直接使用base_size作为“长边”尺寸。
            else:
                longside = self.base_size
            # 这行代码根据图像的长宽比，计算新的高度和宽度。
            # 如果高度大于宽度，则宽度会根据长边长度和原始长宽比进行调整；反之，则高度会根据长边长度进行调整。这样做是为了保持图像的长宽比例。
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            # 这两行代码使用OpenCV的resize函数将图像和标签调整到新的尺寸。图像使用cv2.INTER_LINEAR进行线性插值，适合于缩放时保持图像平滑；
            # 标签使用cv2.INTER_NEAREST进行最近邻插值，这是因为标签是分类的整数值，最近邻插值可以保持这些分类值不发生改变。
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        # 只关心高度和宽度，而不关心通道数，所以使用下划线_来接收通道数的值，表明这个值不会在后续代码中使用，可以忽略。
        h, w, _ = image.shape

        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            #print('111')
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 使用 OpenCV 的 cv2.getRotationMatrix2D 函数生成旋转矩阵。这个函数接受三个参数：旋转的中心点、旋转角度和缩放因子（这里为 1.0，表示不缩放）。
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            # 使用 OpenCV 的 cv2.warpAffine 函数应用旋转矩阵到图像上。
            # 这里指定了旋转矩阵 rot_matrix，输出图像的大小 (w, h)，以及插值方法 cv2.INTER_LINEAR（线性插值，适合于图像内容）。
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)
            # 对标签也应用同样的旋转矩阵，但插值方法是 cv2.INTER_NEAREST（最近邻插值）。最近邻插值是处理标签时的常用选择，因为它不会引入新的标签值，这对于分类标签尤其重要。

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                # 使用 OpenCV 的 copyMakeBorder 函数为图像和标签添加填充。value=0 表示用0值填充。
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            # 随机填充
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
                # 使用 NumPy 的 fliplr 函数进行水平翻转。copy() 用于创建翻转后图像的副本，确保原始图像不被更改。

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            # 计算高斯核的大小（ksize）。核大小与 sigma 成比例，确保模糊效果的自然性。
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            # 确保核大小是奇数，因为高斯模糊的核必须是奇数。
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            # 使用 OpenCV 的 GaussianBlur 函数应用高斯模糊。
            # sigmaX 和 sigmaY 指定水平和垂直方向的标准差，borderType=cv2.BORDER_REFLECT_101 指定边缘处理方式。
        return image, label
        
    def __len__(self):
        # __len__ 方法在 Python 中被用来定义一个对象的“长度”。当你对一个对象使用内置的 len() 函数时，Python 会调用该对象的 __len__ 方法。
        # 在类似于列表、元组、字符串等内置容器中，__len__ 通常返回容器中元素的数量。
        return len(self.files)
        # 当调用 len(dataset)（其中 dataset 是 BaseDataSet 类的一个实例），它会返回数据集中样本的总数。

    def __getitem__(self, index):
        # image, label, image_id = self._load_data(index): 这行调用 _load_data 方法来加载特定索引 (index) 的数据。
        # 这通常包括从磁盘读取图像和相应的标签，并可能还包括一些基本预处理。image_id 可能是该样本的唯一标识符。
        image, label, image_id = self._load_data(index)
        if self.val:
        # if self.val: 和 elif self.augment:: 这两个条件语句用于根据当前模式（验证或训练）应用不同的数据增强。
        # 如果是验证模式 (self.val 为真)，则调用 _val_augmentation 对图像和标签进行处理；
        # 如果是训练模式并且启用了数据增强 (self.augment 为真)，则调用 _augmentation。
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        # 这行代码将标签转换为 PyTorch 张量。首先，使用 np.array 将标签转换为 NumPy 数组，并指定数据类型为 int32。
        # 然后，torch.from_numpy 将 NumPy 数组转换为 PyTorch 张量。.long() 转换张量的数据类型为长整型（64位整数）。
        image = Image.fromarray(np.uint8(image))
        # 这行代码先将图像转换为 uint8 类型的 NumPy 数组，然后使用 Image.fromarray 将其转换为 PIL 图像对象。
        # 这通常用于后续的图像变换或增强。
        if self.return_id:
        # 如果 self.return_id 为真，则返回图像、标签和图像 ID。
        # 首先，self.to_tensor(image) 将 PIL 图像转换为 PyTorch 张量，然后 self.normalize 对张量进行标准化处理（使用在类初始化时定义的均值和标准差）。
            return  self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label
        # 如果 self.return_id 为假，则只返回处理后的图像和标签张量。

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

