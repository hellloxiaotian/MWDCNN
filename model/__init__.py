import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.utils.model_zoo
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args, model=None):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.patch_size = args.patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_GPUs = args.n_GPUs
        self.mode = args.mode
        self.save_models = args.save_models

        if model is None or isinstance(model, str): # 训练时传进来是None/测试时有时候传进来是str（目录）
            module = import_module('model.' + args.model_name.lower())
            self.model, ours = module.make_model(args)
            if ours == 0:
                model = 0
        else: # 如果model不为空，即直接传进来一个可以用的模型
            self.model = model
            print("Model is Created!")

        if self.mode == 'train':
            self.model.train()

            if self.args.pretrain != '':
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(self.args.dir_model, 'pre_train', self.args.model_name, self.args.pre_train)),
                    strict=False)

            # self.model.to(self.device)
            self.model = nn.DataParallel(self.model.to(self.device), device_ids=[i for i in range(self.n_GPUs)])
        elif self.mode == 'test':

            if isinstance(model, str):  # 如果传进来model参数为字符串，表示需要从磁盘加载模型文件
                dict_path = model
                print("Be ready to load model from {}".format(dict_path))

                load_dict = torch.load(dict_path)

                try:
                    self.model.load_state_dict(load_dict, strict=True)
                except RuntimeError:
                    from collections import OrderedDict
                    new_dict = OrderedDict()

                    for key, _ in load_dict.items():    # 去掉开头module.前缀
                        new_dict[key[7:]] = load_dict[key]

                    self.model.load_state_dict(new_dict, strict=True)

                self.model = nn.DataParallel(self.model.to(self.device), device_ids=[i for i in range(self.n_GPUs)])

            self.model.eval()

    def forward(self, x, sigma=None):
        return self.model(x)

