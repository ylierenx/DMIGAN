clc;
clear;
numrows = 1088;
numcols = 1088;
img_size_1 = 1088;
img_size_2 = 1088;
k_1 = 8;
i_m = numrows/32;
j_m = numcols/32;
entropy_img = zeros(1, i_m, j_m);
img_std_pre = 0;
img_std = 0;
for num_ite = 1:20
    rowrank1 = randperm(img_size_1 / k_1);
    a_1 = linspace(1, img_size_1 - k_1 + 1, img_size_1 / k_1);
    a_1 = a_1(rowrank1);
    f_1 = zeros(1, img_size_1);
    f_1 = f_1(1,:);
    for i_1 = 1:(img_size_1 / k_1)
        f_1((i_1-1) * k_1+1) = a_1(i_1);
        for j_1 = 1:k_1-1
            f_1((i_1-1) * k_1+1 + j_1) = a_1(i_1) + j_1;
        end
    end
    rand_sw_p = eye(img_size_1, img_size_1);
    rand_sw_p1 = rand_sw_p(f_1, :);
    rand_sw_p1_t = rand_sw_p1';
    
    rowrank2 = randperm(img_size_2 / k_1);
    a_2 = linspace(1, img_size_2 - k_1 + 1, img_size_2 / k_1);
    a_2 = a_2(rowrank2);
    f_2 = zeros(1, img_size_2);
    f_2 = f_2(1,:);
    for i_2 = 1:(img_size_2 / k_1)
        f_2((i_2-1) * k_1+1) = a_2(i_2);
        for j_2 = 1:k_1-1
            f_2((i_2-1) * k_1+1 + j_2) = a_2(i_2) + j_2;
        end
    end
    rand_sw_p = eye(img_size_2, img_size_2);
    rand_sw_p2 = rand_sw_p(f_2, :);
    rand_sw_p2_t = rand_sw_p2';
    
    %file_path = '.\train_test\';
    f_path_ycbcr = '.\train_test_y\';
    img_path_list = dir(strcat(f_path_ycbcr,'*.tif'));%获取该文件夹中所有jpg格式的图像
    img_num = length(img_path_list);%获取图像总数量
    for j = 1:img_num %逐一读取图像
        
        image_name = img_path_list(j).name;% 图像名
        img = imread(strcat(f_path_ycbcr,image_name));
        im_in = imresize(img, [numrows numcols], 'bicubic');
        im_in_t = double(im_in)*rand_sw_p2;
        im_in_t = rand_sw_p1*im_in_t;
        im_in_t = uint8(im_in_t);
        for a=1:i_m
            for b=1:j_m
                img_temp = im_in_t(((a-1)*32+1):a*32, ((b-1)*32+1):b*32);
                entropy_img(1, a, b) = entropy(img_temp);
            end
        end
        img_std_temp = std2(entropy_img);
        img_std = img_std_temp + img_std;
    end
    if num_ite == 1
        fprintf('%f\n',img_std_pre);
        fprintf('%f\n',img_std);
        fprintf('%f\n',num_ite);
        img_std_pre = img_std;
    end
    if num_ite>1
        if img_std < img_std_pre
            fprintf('%f\n',img_std_pre);
            fprintf('%f\n',img_std);
            fprintf('%f\n',num_ite);
            img_std_pre = img_std;
            
            save('1088.mat', 'rand_sw_p1');
            save('1088_t.mat', 'rand_sw_p1_t');
            %save('rand_sw_p2_mat_train_352_512_rand.mat', 'rand_sw_p2');
            %save('rand_sw_p2_t_mat_train_352_512_rand.mat', 'rand_sw_p2_t');
        end
    end
    img_std = 0;
    %num_ite
        %imwrite(img, 'test.tif');
   
   
    
end
    