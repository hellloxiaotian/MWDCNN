This paper is conducted by Chunwei Tian, Menghua Zheng, Wangmeng Zuo, Bob Zhang, Yanning Zhang, David Zhang. Test codes are released. Also, its training codes are will be released when this paper is accepted. 



## Requirements (Pytorch)

#### Pytorch 1.10.2

#### Python 3.8

#### openCv for Python



## Commands

### Test

cd main

### Color noisy images

python w_test.py --model_file_name ../model_zoo/wmdcnn/c25/model_sigma25.pth --mode test --test_dataset CBSD68
--model_name wmdcnn --rgb_range 1 --n_GPUs 1 --GPU_id 1 --n_colors 3 --sigma 25
