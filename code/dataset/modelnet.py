import os, os.path as osp
import glob
import h5py
import numpy as np
from functools import partial
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
BASE_DIR = osp.dirname(os.path.abspath(__file__))
DATA_DIR = osp.join(BASE_DIR, 'data')


def download_modelnet40():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data_cls(partition):
    download_modelnet40()
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)[:, 0]
    return all_data, all_label


def jitter(x: np.ndarray, sigma=0.01, clip=0.05, p=0.5):
    if np.random.rand() < p:
        return x + np.clip(sigma * np.random.randn(*x.shape), -clip, clip)
    else:
        return x


def rotate(x: np.ndarray, axis='y', angle=15, p=0.5):
    # x: (n, 3)
    # rotate along y-axis (up)
    if np.random.rand() < p:
        R = Rotation.from_euler(axis, np.random.uniform(-angle, angle), degrees=True).as_matrix()
        x = x @ R
        return x
    else:
        return x


def translate(x: np.ndarray, shift=0.2, p=0.5):
    if np.random.rand() < p:
        return x + np.random.uniform(-shift, shift, size=x.shape[-1])
    else:
        return x


def anisotropic_scale(x: np.ndarray, min_scale=0.66, max_scale=1.5, p=0.5):
    # x: (n, 3)
    if np.random.rand() < p:
        scale = np.random.uniform(min_scale, max_scale, size=x.shape[-1])
        return x * scale
    else:
        return x


class ModelNet40(Dataset):
    def __init__(self,
                 train=True,
                 rotate_func=partial(rotate, axis='y', angle=15, p=0.5),
                 jitter_func=partial(jitter, sigma=0.01, clip=0.05, p=0.5),
                 translate_func=partial(translate, shift=0.2, p=0.5),
                 anisotropic_scale_func=partial(anisotropic_scale, min_scale=0.66, max_scale=1.5, p=0.5),
                 ):
        self.train = train
        self.data, self.label = load_data_cls('train' if train else 'test')

        self.rotate = rotate_func
        self.jitter = jitter_func
        self.translate = translate_func
        self.anisotropic_scale = anisotropic_scale_func

    def __getitem__(self, item):
        pcd = self.data[item]
        label = self.label[item]
        if self.train:
            pcd = self.rotate(pcd)
            pcd = self.jitter(pcd)
            pcd = self.translate(pcd)
            pcd = self.anisotropic_scale(pcd)
            np.random.shuffle(pcd)
        return pcd.astype('f4'), label.astype('i8')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    load_data_cls('train')
    load_data_cls('test')
