import os
import pandas as pd
import cv2
import numpy as np
import argparse
from dandelion.ext.CV import CV
import multiprocessing

def get_image_list_and_targets(dirpath, image_path):
    train_data = pd.read_csv(dirpath)
    image_list = []
    for image in train_data['Id']:
        path_1 = os.path.join(image_path, image + '_green' + '.png')
        path_2 = os.path.join(image_path, image + '_blue' + '.png')
        path_3 = os.path.join(image_path, image + '_red' + '.png')
        path_4 = os.path.join(image_path, image + '_yellow' + '.png')

        paths = [path_1, path_2, path_3, path_4]
        image_list.append(paths)

    labels = []
    for label in train_data['Target']:
        label = label.split(' ')
        sublabel = []
        for item in label:
            sublabel.append(int(item))
        labels.append(sublabel)

    return image_list, labels

def get_single_data(file_list, npy_list, labels, data_block_pos, return_dict, image_size=[512, 512], channel=3, dtype='float32'):
    file_num = len(file_list)
    npy_num = len(npy_list)
    if npy_num != file_num:
        print('file num and npy num error!')

    # data_blob = np.zeros(shape=(1, channel, image_size[0], image_size[1]), dtype=dtype)
    error_list = []

    for i in range(file_num):
        try:
            dandlion_cv_out_g = CV.imread(file_list[i][0], flatten=False)
            dandlion_cv_out_b = CV.imread(file_list[i][1], flatten=False)
            dandlion_cv_out_r = CV.imread(file_list[i][2], flatten=False)
            dandlion_cv_out_y = CV.imread(file_list[i][3], flatten=False)
        except BaseException:
            # print('%s read error' % file_list[i])
            error_list.append(file_list[i])
            continue

        green = np.zeros(shape=(1, image_size[0], image_size[1], channel), dtype=dtype)
        blue = np.zeros(shape=(1, image_size[0], image_size[1], channel), dtype=dtype)
        red = np.zeros(shape=(1, image_size[0], image_size[1], channel), dtype=dtype)
        yellow = np.zeros(shape=(1, image_size[0], image_size[1], channel), dtype=dtype)

        # dandlion_cv_out = CV.imresize(dandlion_cv_out, image_size)
        green[0, :, :, :] = dandlion_cv_out_g[:, :, :]
        blue[0, :, :, :] = dandlion_cv_out_b[:, :, :]
        red[0, :, :, :] = dandlion_cv_out_r[:, :, :]
        yellow[0, :, :, :] = dandlion_cv_out_y[:, :, :]
        label = labels[i]
        data_blob = {'green': green, 'blue': blue, 'red': red, 'yellow': yellow, 'label': label}
        np.save(npy_list[i], data_blob)

        if i % 100 == 0:
            print('data block = %d, Read Num = %d' % (data_block_pos, i))

    if len(error_list) != 0:
        return_dict[data_block_pos] = error_list


if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-dirpath', default=r'C:\Users\Classiclaw\Documents\JoeJia\kaggle\data\train',type=str)
    argparser.add_argument('-csvpath', default=r'C:\Users\Classiclaw\Documents\JoeJia\kaggle\data\train_test.csv')
    argparser.add_argument('-output_prefix',default=r'C:\Users\Classiclaw\Documents\JoeJia\kaggle\data\test_npy', type=str)
    argparser.add_argument('-num_per_npy', default=100, type=int)  # mode 1: decide process num
    arg = argparser.parse_args()

    dirpath = arg.dirpath
    csvpath = arg.csvpath
    output_prefix = arg.output_prefix
    num_per_npy = arg.num_per_npy

    image_list, labels = get_image_list_and_targets(csvpath, dirpath)
    samples_num = len(image_list)
    print('samples num: %d' % (samples_num))

    npy_list = []
    for i in range(samples_num):
        npy_name = os.path.join(output_prefix, str(i) + '.npy')
        npy_list.append(npy_name)

    block_num = samples_num // num_per_npy + 1

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    for k in range(block_num):
        start = k * num_per_npy
        end = min((k + 1) * num_per_npy, samples_num)
        p = multiprocessing.Process(target=get_single_data,
                                    args=(image_list[start:end], npy_list[start:end], labels[start:end], k, return_dict, [512, 512]))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    # ------------------ delete the invalid file in image and npy list -------------------#
    # print('read error image num: %d' % (len(return_dict.values())))
    if len(return_dict.values()) != 0:
        err_list = []
        for err in return_dict.values():
            err_list.extend(err)
        print('read error image num: %d' % (len(err_list)))
        err_index = []
        for err in err_list:
            index = image_list.index(err)
            err_index.append(index)
        print(err_index)

        err_index.sort()
        i = 0
        for idx in err_index:
            idx = idx - i
            image_list.pop(idx)
            npy_list.pop(idx)
            i += 1

        file_name = 'read_err_0' + ".list"
        list_name = os.path.join(output_prefix, file_name)
        with open(list_name, 'w+') as err_file:
            for i in err_list:
                err_file.writelines(i)
                err_file.write('\n')

    valid_file_num = len(image_list)
    print('valid image num: %d' % (valid_file_num))

    # --------------------- write the image list and npy list file ---------------------- #
    # file_name = 'filepath_0' + ".list"
    # list_name = os.path.join(output_prefix, file_name)
    # with open(list_name, 'w+') as list_file:
    #     for i in range(valid_file_num):
    #         list_file.writelines(image_list[i])
    #         list_file.write('\n')

    file_name = 'npypath_0' + ".list"
    npy_name = os.path.join(output_prefix, file_name)
    with open(npy_name, 'w+') as list_npy:
        for i in range(valid_file_num):
            list_npy.writelines(npy_list[i])
            list_npy.write('\n')

    # print(err_index)
    print('Done!')


