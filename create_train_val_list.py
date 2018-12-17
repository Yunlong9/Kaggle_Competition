import os
import numpy as np

RATE = 0.8

def get_all_file_list(dirpath):
    file_list = []
    with open(dirpath, mode='rt', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            file_list.append(line)

    return file_list

if __name__=='__main__':
    dirpath = r'C:\Users\Classiclaw\Documents\JoeJia\kaggle\data\test_npy\npypath_0.list'
    output_path = r'C:\Users\Classiclaw\Documents\JoeJia\kaggle\data\test_npy'

    npy_list = get_all_file_list(dirpath)
    samples_num = len(npy_list)
    print('total samples num: %d' % samples_num)

    file_idx = list(range(samples_num))
    np.random.shuffle(file_idx)

    train_samples_num = int(samples_num * RATE)

    train_list = file_idx[: train_samples_num]
    val_list = file_idx[train_samples_num :]
    # print(train_list)
    # print(val_list)
    train_name = 'train_files.list'
    train_path = os.path.join(output_path, train_name)
    with open(train_path, mode='w+') as train_f:
        for index in train_list:
            train_f.writelines(npy_list[index])
            train_f.write('\n')

    test_name = 'val_files.list'
    test_path = os.path.join(output_path, test_name)
    with open(test_path, mode='w+') as test_f:
        for index in val_list:
            test_f.writelines(npy_list[index])
            test_f.write('\n')
