import numpy as np

def get_all_file_list_from_dirfile(dirfile):
    file_list = []
    with open(dirfile, mode='rt', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            file_list.append(line)

    return file_list

def get_one_hot_label(label, Nclass=28):
    one_hot_label = np.zeros((Nclass, ), dtype='int32')
    for i in label:
        one_hot_label[i] = 1

    return one_hot_label

if __name__=='__main__':
    labels = [2, 7]
    one_hot_label = get_one_hot_label(labels, 28)
    print(one_hot_label)

    one_hot_labels = np.zeros(shape=(2, 28), dtype='int32')
    one_hot_labels[1, :] = one_hot_label
    print(one_hot_labels)
