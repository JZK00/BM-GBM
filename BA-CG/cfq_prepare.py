import glob

import pandas as pd
import os

if __name__ == "__main__":
    ori_path = r"C:\Users\wd976311\OneDrive - Varian Medical Systems\Desktop\Projects\varian project\xiangya\XiangYa\code\data_analysis"
    data_path = r'C:\Users\wd976311\OneDrive - Varian Medical Systems\Desktop\Projects\varian project\xiangya\XiangYa\HNSCC_data\correction'
    orinames = pd.read_csv(os.path.join(ori_path, 'training_names.csv'))
    newpd = pd.DataFrame(columns=['Image', 'Mask', 'label'])
    data_qianzui = 't2'
    label_qianzui = 'T2'
    labels = pd.read_csv(os.path.join(ori_path, 'target.csv'))

    for index, name in enumerate(orinames.name.values):
        print(name)
        label = labels[labels['Number']==name]['ki67_express'].values[0]
        if name.startswith('E'):
            files = os.path.join(data_path, name)
        else:
            files = os.path.join(data_path, name.zfill(10))
        assert len(os.listdir(files)) > 0
        Imagepath = os.path.join(files, '{}_correction.nii.gz'.format(data_qianzui))
        Maskpath = os.path.join(files, '{}_label.nii.gz'.format(label_qianzui))
        newpd.loc[index] = [Imagepath, Maskpath, label]

    newpd.to_csv('./data/{}_feature.csv'.format(data_qianzui))
