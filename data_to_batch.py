import numpy as np
import psutil
import os
from data_process import get_one_hot_label
from data_process import get_all_file_list_from_dirfile

def load_batch_data_from_harddisk(all_npy_list, batch_index_list, imsize=(512, 512), channel=3, data_ty='float32'):
    batch_size = len(batch_index_list)
    batch_index_new = batch_index_list

    X = np.zeros(shape=(batch_size, imsize[0], imsize[1], channel * 4), dtype=data_ty)
    Y = np.zeros(shape=(batch_size, 28), dtype='int32')
    i = 0
    for file_idx in batch_index_new:
        data = np.load(all_npy_list[i])
        data = data.item()
        Y[i] = get_one_hot_label(data['label'])

        green = data['green']
        blue = data['blue']
        red = data['red']
        yellow = data['yellow']

        four_image = np.concatenate((green, blue, red, yellow), axis=3)

        I = (255.0 - four_image) / 255.0
        X[i, :, :, :] = I
        i += 1

    return X, Y


def get_batch_from_harddisk(all_npy_list, index_list, data_queue=None, max_epoch=10000, batch_size=16, imsize=(512, 512),
                      shuffle=False, supplement=False, channel=3):
    me_process = psutil.Process(os.getpid())
    sample_num = len(index_list)

    for epoch in range(max_epoch):
        if me_process.parent() is None:     # parent process is dead
            raise RuntimeError('Parent process is dead, exiting')

    batch_num = sample_num // batch_size
    if supplement:
        if sample_num / batch_size > batch_num:
            batch_num += 1

    if shuffle:
        index = np.random.choice(index_list, size=sample_num, replace=False)
    else:
        index = index_list

    for batch_idx in range(batch_num):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, sample_num)

        batch_file_idx = index[start_idx:end_idx]
        # batch_file_list = [all_file_list[idx] for idx in batch_file_idx]
        if end_idx < (batch_idx + 1) * batch_size:
            for i in range((batch_idx + 1) * batch_size - sample_num):
                idx = np.random.choice(index_list, size=1, replace=False)
                batch_file_idx.append(idx)

        X, Y = load_batch_data_from_harddisk(all_npy_list, batch_file_idx, imsize, channel)
        data_queue.put((X, Y))

if __name__=='__main__':
    dirfile = r'C:\Users\Classiclaw\Documents\JoeJia\kaggle\data\test_npy\npypath_0.list'
    all_npy_list = get_all_file_list_from_dirfile(dirfile)
    # print(all_npy_list)
    batch_list = [0, 2, 4, 6]
    X, Y = load_batch_data_from_harddisk(all_npy_list, batch_list)
    print(X.shape)
    print(Y.shape)
    print(Y)