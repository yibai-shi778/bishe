import csv
import sys
import os

import numpy as np


def labels_pre_processing(label_path, relabel_path):
    with open(label_path, 'r') as r:
        label_list = np.array(list(csv.reader(r)))
        id_list = label_list[:, 0]
        engagement_list = label_list[:, 2]
        for i in range(1, len(id_list)):
            if int(engagement_list[i]) > 1:
                engagement_list[i] = 1
            else:
                engagement_list[i] = 0
    with open(relabel_path, 'w+', newline='') as w:
        writer = csv.writer(w)
        for row in zip(id_list, engagement_list):
            writer.writerow(row)
        w.close()


if __name__ == '__main__':

    cur_dir = sys.path[1]
    labels_pre_processing(os.path.join(cur_dir, 'labels\\TestLabels.csv'),
                          os.path.join(cur_dir, 'labels\\ReTestLabels.csv'))
