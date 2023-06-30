# DMIGAN

这是发表于电子学报的论文《基于MAP的多信息流梯度更新与聚合视频压缩感知重构算法》的pytorch实现，作者为杨鑫、杨春玲。

## 各文件说明

（1）check_point文件夹下保存了各个采样率的训练模型 

（2）permutation_matrix文件夹下保存了各置换矩阵

（3）permutation_matrix_generation.m文件为生成置换矩阵的matlab代码，对应论文中的基于图像信息熵的块置换矩阵选择算法

（4）dataloder1.py文件为关键帧网络训练时的数据加载代码

        1、gop_size设为1，表示每次训练只取一个序列中的随机一帧
        
        2、image_size表示图像块尺寸
        
        3、load_filename参数表示训练数据的参数路径列表
        
（5）dataloder2.py文件为非关键帧网络训练时的数据加载代码

        1、gop_size表示GOP大小+1，如果是8，则该值设成9
        
        2、image_size表示图像块尺寸
        
        3、load_filename参数表示训练数据的参数路径列表
        
（6）fnames_shuffle_test.npy、fnames_shuffle_train.npy与fnames_shuffle_val.npy表示UCF-101数据集的训练、测试与验证集数据的数据路径列表

（7）model_dmigan.py文件为各个模型实现代码

        1、num_of、num_of1和num_of2分别表示整体假设集、参考帧1假设集和参考帧2假设集假设数
        
        2、backWarp_MH为多假设反向对齐
        
        3、gd_fusion与final_fusion分别为梯度融合与最终融合网络
        
        4、denoising为去噪残差块
        
        5、spynet_mae_mh为基于spynet的多假设光流估计
        
        6、basic_block为基础级联模块，包含多信息流梯度更新实现
        
（8）test_dmigan.py文件为测试代码

        1、rgb表示是否测试彩色图像，是则设为True
        
        2、flag表示是否加载训练好的模型，设为True
        
        3、block_size表示采样块尺寸
        
        4、gop_size表示GOP大小
        
        5、image_width与image_height表示当分辨率无法整除采样块大小时，进行0填充，填完以后的图像尺寸
        
        6、img_w与img_h表示原图尺寸
        
        7、num_gop为GOP数目
        
        8、test_video_name为测试文件夹，可放在当前py文件同级目录下
        
        9、sr表示非关键帧采样率
        
（9）train_key_hsganet.py为关键帧训练代码，为对hsganet的重新实现，代码更加清晰

（10）train_nonkey_map.py为非关键帧训练代码

## Check_point_path

训练好模型的下载链接为：(https://drive.google.com/drive/folders/1Iroa4g9kmtlaCD7KSE8GuMXGbGpgV7S0)


