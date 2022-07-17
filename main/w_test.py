import glob
import sys
import cv2
# from skimage.measure import compare_psnr
from torch import nn

sys.path.append('../')
import os
import torch.nn.functional
import numpy as np
import torch.optim as optim
from option import args
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tool.MyDataSet import Art_nosie_Dataset, Multi_Art_nosie_Dataset
from tool.MyDataSet import Real_Dataset
from tool.common_tools import ModelTrainer
from tool.common_tools import save_to_file
from tool.common_tools import batch_PSNR
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import image_similarity_measures
from image_similarity_measures.quality_metrics import fsim, psnr, ssim
from tqdm import tqdm
import random
import time

seed = args.seed

if args.n_colors == 3:
    color = 'color'

    if args.sigma == 200.0:  # 使用的是真实噪声
        print("Use the realnoise dataset...........")
        My_Dataset = Real_Dataset

        args.train_dataset = "real_dataset/SSID+realworld/"  # 训练集只有有一个
        args.test_dataset = "real_dataset"  # 测试集也只有一个
        print("n_color is 3, Training in real_dastaset, evaluate in realdataset")

    else:   # 使用的是 人工噪声
        My_Dataset = Art_nosie_Dataset
        # args.train_dataset = "CBSD432+pristine_images"  # 训练集已有一个

        # args.train_dataset = "CBSD432+pristine+DIV2k+Flick2k"
        args.train_dataset = "CBSD432"

        # 测试集有多个
        if args.test_dataset == "CBSD68": # CBSD68是默认值
            print("n_color is 3, Training in CBSD432, evaluate in CBSD68")
        else:
            print("n_color is 3, Training in CBSD432, evaluate in {}".format(args.test_dataset))

elif args.n_colors == 1:
    color = 'g'
    args.train_dataset = "BSD432+pristine+DIV2K+Flick2k"
    # args.train_dataset = "BSD432"

    if args.model_name == 'mstd':
        My_Dataset = Multi_Art_nosie_Dataset
        # My_Dataset = Art_nosie_Dataset
    else:
        My_Dataset = Art_nosie_Dataset
        print('My Dataset choose Art_nosie_Dataset')

    if args.test_dataset == "CBSD68":  # BSD68是默认值
        args.test_dataset = "BSD68"
        print("n_color is 1, Training in %s , evaluate in %s" % (args.train_dataset, args.test_dataset))
    else:
        print("n_color is 1, Training in BSD432, evaluate in {}".format(args.test_dataset))
else:
    raise ValueError("args.n_color must equal 1 or 3 in interage")


train_dir = args.dir_data + 'train/' + args.train_dataset
test_dir = args.dir_data + 'test/' + args.test_dataset
save_model_dir = args.save_base + os.path.join(args.dir_model)
save_state_dir = args.save_base + os.path.join(args.dir_state)
save_loss_dir = args.save_base + os.path.join(args.dir_loss)
save_test_dir = args.save_base + os.path.join(args.dir_test_img)
tensorboard_dir = args.save_base + os.path.join(args.dir_tensorboard)


os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_id


def log(*args, **kwargs):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def test(args, data_loader, save_test_dir, save=False, model_file=None, loss_f=None):
    args.mode = 'test'

    if save:
        torch.manual_seed(1234)

    import model
    # _model = model.Model(args, model_file)
    _model = torch.load(model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log('load trained model to GPU')

    psnrs = []
    ssims = []
    fsims = []
    L = []
    save_dict = {}
    psnr_dict = {}
    ssim_dict = {}

    # for n_count, data in enumerate(data_loader):
    torch.manual_seed(1234)
    for data in tqdm(data_loader):
        # 取出数据和标签
        ori_img, nos_img, img_name = data

        img_name = img_name[0]
        noise = torch.FloatTensor(ori_img.size()).normal_(mean=0, std=args.sigma / 255.)
        # nos_img = noise + ori_img

        # 将数据和标签传到GPU上

        with torch.no_grad():

            nos_img = nos_img.to(device)

            if args.model_name == 'ffdnet':
                # print("The model is ffdnet now!")
                sigma = torch.full((1, 1, 1, 1), args.sigma * args.rgb_range / 255.).type_as(nos_img)
                output = _model(nos_img, sigma)
            else:
                output = _model(nos_img)

        output = output.cpu()
        nos_img = nos_img.cpu()

        psnr_x_ = batch_PSNR(output, ori_img, 1.)
        if isinstance(loss_f, tuple):
            loss_ = loss_f[0](output, ori_img)+loss_f[1](output, ori_img)
        else:
            loss_ = loss_f(output, ori_img)

        # 调整两张图片维度等信息
        output = output.squeeze_()      # 最初始的output是一个4维的tensor，将其压缩至3维
        ori_img = ori_img.squeeze_()    # 最初始的ori_img是一个4维的tensor，将其压缩至3维
        nos_img = nos_img.squeeze_()    # 最初始的nos_img是一个4维的tensor，将其压缩至3维

        # loss_ = loss_f(output, ori_img) / 2.0

        output = output.mul_(255. / args.rgb_range)  # 将输出值（0~1 or 0~255）转化成像素（0~255）
        ori_img = ori_img.mul_(255. / args.rgb_range)
        nos_img = nos_img.mul_(255. / args.rgb_range)

        if args.n_colors == 3: # 彩色图像需要调整，灰度图像不需要调整
            output = output.permute(1, 2, 0)        # chw --> hwc
            ori_img = ori_img.permute(1, 2, 0)      # chw --> hwc
            nos_img = nos_img.permute(1, 2, 0)      # chw --> hwc
            # nos_img = nos_img.view(nos_img.shape[1], nos_img.shape[2], -1)  # chw --> hwc

        np_output = np.uint8(output.detach().clamp(0, 255).round().numpy())         # tensor --> np(intenger)
        np_img_rgb = np.uint8(ori_img.detach().clamp(0, 255).round().numpy())       # tensor --> np(intenger)
        np_img_nos = np.uint8(nos_img.detach().clamp(0, 255).round().numpy())       # tensor --> np(intenger)

        # 保存结果图，先存储到字典中
        save_dict['denoi_'+img_name] = np_output

        psnr_x_ = peak_signal_noise_ratio(np_output, np_img_rgb)
        ssim_x_ = structural_similarity(np_img_rgb, np_output, multichannel=True)
        # ssim_x_ = ssim(np_output, np_img_rgb)

        if args.n_colors == 3:
            fsim_x_ = fsim(np_img_rgb, np_output)
        else:
            re_img = np.transpose(np.repeat(np.expand_dims(np_img_rgb, 0), 3, 0), (1, 2, 0))
            re_out = np.transpose(np.repeat(np.expand_dims(np_output, 0), 3, 0), (1, 2, 0))
            fsim_x_ = fsim(re_img, re_out)

        psnr_dict[img_name] = psnr_x_
        ssim_dict[img_name] = ssim_x_

        psnrs.append(psnr_x_)
        ssims.append(ssim_x_)
        fsims.append(fsim_x_)
        L.append(loss_.item())

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    fsim_avg = np.mean(fsims)
    loss_avg = np.mean(L)/2

    psnr_max = np.max(psnrs)
    ssim_max = np.max(ssims)

    if save: # 保存图片和loss
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M')

        #创建文件夹
        save_path = os.path.join(save_test_dir, args.model_name, color, str(args.sigma), args.test_dataset, str(args.flag))

        print(save_path)

        if not os.path.exists(save_path):   # 没有该文件夹，则创建该文件夹
            os.makedirs(save_path)
            print("Make the dir:{}".format(save_path))

        # 保存相关信息
        # f = open(os.path.join(save_path, 'test_result.txt'), 'w')
        # f.close()
        txtname = os.path.join(save_path, 'test_result.txt')
        if not os.path.exists(txtname):
            os.system(r"touch {}".format(txtname))

        save_to_file(os.path.join(save_path, 'test_result.txt'),
                     "Time: {}, psnr_avg: {:.4f},  ssim_avg: {:.4f}\npsnr_max: {:.4f}, ssim_max:{:.4f} \nfsim_avg:{:.4f}\n".\
                     format(time_str, psnr_avg, ssim_avg, psnr_max, ssim_max, fsim_avg))

        # 保存命令行参数
        p_str = " ".join(sys.argv)
        save_to_file(os.path.join(save_path, 'test_result.txt'), p_str+'\n')

        # 保存每张图片的psnr,ssim
        for k1, k2 in zip(psnr_dict, ssim_dict):
            save_to_file(os.path.join(save_path, 'test_result.txt'),
                         "{}, psnr: {}, ssim: {}\n".format(k1, psnr_dict[k1], ssim_dict[k2]))
        # 保存图片
        for k in save_dict:
            img = cv2.cvtColor(save_dict[k], cv2.COLOR_RGB2BGR)  # RGB->BGR
            cv2.imwrite(os.path.join('output', k[:-3]+'png'), img)

    print("Psnr_Avg:{:.2f}, ssim_avg:{:.4f} \nPsnr_Max: {:.2f}, ssim_max:{:.4f} \nFsim_avg:{:.4f} \nLoss_avg:{:.4f}".\
          format(psnr_avg, ssim_avg, psnr_max, ssim_max, fsim_avg, loss_avg))




    return psnr_avg, ssim_avg, loss_avg


if __name__ == '__main__':
    if args.mode == 'train':
        print("Start to train.......")
        # train(args)
    elif args.mode == 'test':
        print("Start to test.......")
        # test_data = DNCNNs_Dataset(args, data_dir=test_dir, mode='test')
        # test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

        test_data = My_Dataset(args, data_dir=test_dir, mode='test')
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        criterion = nn.MSELoss(reduction='sum')
        psnr_avg, ssim_avg, _loss = test(args,
                                      test_loader,
                                      save_test_dir,
                                      save=True,
                                      model_file=args.model_file_name,
                                      loss_f=criterion)

    elif args.mode == 'inference':
        print("Start to inference.......")
        pass