# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/
# @date       : 2021-07-03
# @brief      : 通用函数

    #class
    |
    ——class ModelTrainer(object)
        |
        ——def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
    |
    ——class BatchRename()
        |
        ——def rename(self)


    #function
    |
    ——def process_img(path_img)
    ——def show(x, title=None, cbar=False, figsize=None)
"""
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

import os


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, writer=None):
        loss_sigma = []

        for n_count, data in enumerate(data_loader):

            # 取出数据和标签
            ori_img, nos_img = data
            # 将数据和标签传到GPU上

            ori_img, nos_img = ori_img.to(device), nos_img.to(device)

            # 获得output
            outputs = model(nos_img)
            # 在计算下下一个min-batch之前，先将上一个batch的梯度清零
            optimizer.zero_grad()

            # 计算损失函数
            # loss = loss_f(outputs, ori_img)
            if isinstance(outputs, tuple):
                loss = (loss_f(outputs[0], ori_img) + loss_f(outputs[1], ori_img) + loss_f(outputs[2], ori_img)) / (ori_img.size()[0] * 2)
            else:
                if isinstance(loss_f, tuple):
                    loss = (loss_f[0](outputs, ori_img) + loss_f[1](outputs, ori_img))/(ori_img.size()[0] * 2)
                else:
                    loss = loss_f(outputs, ori_img) / (ori_img.size()[0] * 2)

            # 损失函数计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计loss
            loss_sigma.append(loss.item())

            # 每2k个iteration 打印一次训练信息，loss为200个iteration的平均
            if n_count % 2000 == 2000 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                    epoch_id + 1, max_epoch, n_count + 1, len(data_loader), np.mean(loss_sigma)))

            if writer is not None:
                writer.add_scalars("Loss by iters", {"Loss_train": loss.item()}, n_count + epoch_id * len(data_loader))

        return np.mean(loss_sigma)


def process_img(args, img_rgb):

    # img --> nosing_img
    img_nosing = add_noise(img_rgb, args.sigma)

    # img --> np.array
    img_nos_np = np.array(img_nosing, dtype=np.float32) / 255.0
    # img_np --> tensor and chw --> bchw
    img_nos_tensor = torch.from_numpy(img_nos_np).view(1, -1, img_nos_np.shape[0], img_nos_np.shape[1])

    return img_nos_tensor, img_nosing, img_rgb


class BatchRename():

    def __init__(self):
        self.path = r'../data/images/test'  # 表示需要命名处理的文件夹

    def rename(self):
        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(self.path)
        total_num = len(filelist) # 获取文件夹内所有文件个数
        i = 1  # 表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.jpg'):
                # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的即可）

                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path),str(i) + '.jpg')

                # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                # 这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式

                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue

        print ('total %d to rename & converted %d jpgs' % (total_num, i))


def save_to_file(pth, contents):
    fh = open(pth, 'a')
    fh.write(contents)
    fh.close()


def add_noise(img, noise_leve=25, rgb_range=255):
    # 产生噪声
    noise = torch.randn(img.size()).mul_(noise_leve * rgb_range / 255.)
    # noise = torch.FloatTensor(img.size()).normal_(mean=0, std=noise_leve * rgb_range / 255.)
    # 将噪声加入图片中
    noise_hr = (noise + img).clamp(0, rgb_range)

    return noise_hr


# 将灰色分割图像转变成彩色分割图像
def seg_gray2color(g_path, image_path):
    # 打开两张图片
    gray_seg = cv2.imread(g_path, 0)
    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gray_seg.astype()


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        # PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])


if __name__=='__main__':
    demo = BatchRename()
    demo.rename()