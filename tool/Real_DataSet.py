import sys
sys.path.append('../')
import os
import numpy as np
from utils import utils_image
import torchvision.transforms as transforms
from itertools import chain
from torch.utils.data import Dataset
from tool.common_tools import add_noise
import random


class Real_Dataset(Dataset):    # 真实噪声

    def __init__(self, args, data_dir, mode='train'):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        """
        self.args = args
        self.mode = mode
        self.patch_size = args.patch_size
        # self.nose_level = args.sigma # 采用真实噪声，不需要添加高斯噪声
        self.n_patches = args.n_pat_per_image
        self.n_channels = args.n_colors
        if self.mode == 'train':
            self.gt_lsit, self.real_list = self.train_data_generator(data_dir)  # patches 的集合
        elif self.mode == 'test':
            self.gt_lsit, self.real_list = self.test_data_generator(data_dir)  # patches 的集合

    def __getitem__(self, index):
        clean_img = self.gt_lsit[index]
        real_img = self.real_list[index]

        # ---------------------------------------
        # HWC to CHW, numpy(uint) to tensor
        # ---------------------------------------

        clean_img = utils_image.uint2tensor3(clean_img).mul(self.args.rgb_range/255.)
        real_img = utils_image.uint2tensor3(real_img).mul(self.args.rgb_range / 255.)

        # add noise
        '''
        if self.nose_level != 100:  # 噪声水平确定
            nos_img = add_noise(clean_img, noise_leve=self.nose_level, rgb_range=self.args.rgb_range)
        else:  # nose_level == 100 表示 盲降噪
            nos_img = add_noise(clean_img, noise_leve=np.random.randint(0, 50, 1)[0], rgb_range=self.args.rgb_range)
        '''

        return clean_img, real_img

    def __len__(self):
        return len(self.gt_lsit)

    def gen_patches(self, img1, img2, patch_size=48, n=128, aug=True, aug_plus=False):

        '''
        :param img: input_img
        :param patch_size:
        :param n: a img generate n patches
        :param aug: if need data augmentation or not
        :return: a list of patches
        '''

        patches1 = list()
        patches2 = list()

        ih, iw, _ = img1.shape

        ip = patch_size

        for _ in range(0, n):   # 一张图片产生n个patches
            iy = random.randrange(0, ih - ip + 1)
            ix = random.randrange(0, iw - ip + 1)

            # --------------------------------
            # get patch
            # --------------------------------
            patch1 = img1[iy:iy + ip, ix:ix + ip, :]
            patch2 = img2[iy:iy + ip, ix:ix + ip, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            if aug:  # need augmentation
                if aug_plus:
                    mode = random.randint(0, 6)
                    f_aug = utils_image.augment_img_plus
                else:
                    mode = random.randint(0, 7)
                    f_aug = utils_image.augment_img
            else:  # don't need augmentation
                mode = 0
                f_aug = utils_image.augment_img

            patch1 = f_aug(patch1, mode=mode)
            patch2 = f_aug(patch2, mode=mode)

            patches1.append(patch1)
            patches2.append(patch2)

        return patches1, patches2

    def train_data_generator(self, data_dir):
        # 用于存储所有图片
        real_img_list = list()
        gt_img_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        reallist = os.listdir(os.path.join(data_dir,"real"))
        gtlist = os.listdir(os.path.join(data_dir, "mean"))

        for realimage, gtimage in reallist, gtlist:

            real_path = os.path.join(data_dir, "real", realimage)
            gt_path = os.path.join(data_dir, "mean", gtimage)

            # 打开一张图片
            real_img = utils_image.imread_uint(real_path, n_channels=self.n_channels)  # RGB
            gt_img = utils_image.imread_uint(gt_path, n_channels=self.n_channels)  # RGB

            # 1张图片产生Patches

            real_patches = self.gen_patches(real_img, patch_size=self.patch_size, n=self.n_patches,
                                            aug_plus=self.args.aug_plus)
            gt_patches = self.gen_patches(gt_img, patch_size=self.patch_size, n=self.n_patches,
                                            aug_plus=self.args.aug_plus)
            # 将patches加入到img_lsit中
            real_img_list.append(real_patches)
            gt_img_list.append(gt_patches)

        # img_lsit 为所有patches的集合
        real_img_list = list(chain(*real_img_list))
        gt_img_list = list(chain(*gt_img_list))

        return gt_img_list, real_img_list

    def test_data_generator(self, data_dir):
        # 用于存储所有图片
        real_img_list = list()
        gt_img_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        reallist = os.listdir(os.path.join(data_dir, "real"))
        gtlist = os.listdir(os.path.join(data_dir, "mean"))

        for realimage, gtimage in reallist, gtlist:
            real_path = os.path.join(data_dir, "real", realimage)
            gt_path = os.path.join(data_dir, "mean", gtimage)

            # 打开一张图片
            real_img = utils_image.imread_uint(real_path, n_channels=self.n_channels)  # RGB
            gt_img = utils_image.imread_uint(gt_path, n_channels=self.n_channels)  # RGB

            '''
            # 1张图片产生Patches
            real_patches = self.gen_patches(real_img, patch_size=self.patch_size, n=self.n_patches,
                                            aug_plus=self.args.aug_plus)
            gt_patches = self.gen_patches(gt_img, patch_size=self.patch_size, n=self.n_patches,
                                          aug_plus=self.args.aug_plus)
            '''

            # 将patches加入到img_lsit中
            real_img_list.append(real_img)
            gt_img_list.append(gt_img)

        return gt_img_list, real_img_list
