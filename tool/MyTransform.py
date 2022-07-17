import numpy as np
import torch
from PIL import Image


class AddGaussianNoise(object):

    def __init__(self, sigma=15.0):
        self.sigma = sigma

    def __call__(self, img):
        np_img = np.array(img)
        # 产生噪声
        noise = torch.randn(np_img.shape).mul_(self.sigma)
        np_noise = noise.detach().numpy()

        # 噪声和原始图片叠加
        np_img = np_noise + np_img
        np_img[np_img > 255] = 255  # 避免有值超过255而反转

        img = Image.fromarray(np_img.astype('uint8')).convert('RGB')

        return img


if __name__ == '__main__':
    import torchvision.transforms as transforms

    img = Image.open('../data/images/train/3.jpg')
    to_tensor =transforms.ToTensor()
    tensor_img = to_tensor(img)
    np_img = np.array(img)
    # np_img = np_img.astype(np.float64)

    or_img = np.array(np_img, dtype=np.float32)
    or_img = torch.from_numpy(or_img).view(1, -1, or_img.shape[0], or_img.shape[1])

    tensor_img_2 = torch.from_numpy(np_img)

    # print(or_img.shape)     # 查看图片尺寸
    for i in range(1,2):
        print(i)