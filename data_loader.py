import glob
import os

import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def get_data_loader(data_path, opts):
    """Create training and test data loaders."""
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'advanced':
        # 进阶数据增强: 用于缓解判别器在小数据集上的过拟合
        # 1) 先 Resize 到稍大尺寸 (1.1 倍), 然后随机裁剪回目标尺寸, 模拟随机平移
        # 2) 随机水平翻转, 增加样本多样性
        # 3) 颜色抖动 (亮度/对比度/饱和度) 进一步扩充分布
        # 4) 最后 ToTensor + Normalize 到 [-1, 1], 与 tanh 输出对齐
        load_size = int(1.1 * opts.image_size)
        train_transform = transforms.Compose([
            transforms.Resize([load_size, load_size], Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = CustomDataSet(
        os.path.join('data/', data_path), opts.ext, train_transform
    )
    dloader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    return dloader
