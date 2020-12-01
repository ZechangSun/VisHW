import numpy as np
from warnings import filterwarnings
import argparse
import csv
filterwarnings('ignore')


def load_data():
    img = np.load('../data/sampled_image.npy')
    label = np.load('../data/sampled_label.npy')
    l, w = img.shape[1], img.shape[2]
    img = img.reshape((-1, l*w)).T
    return img, label, l, w


def PCA(img, dim=None):
    m = np.mean(img, axis=1).reshape((-1, 1))
    nimg = img - m
    cov = nimg@nimg.T
    eigs, eigv = np.linalg.eig(cov)
    norm = eigs@np.conjugate(eigs)
    print('reconstruction rate: %.4f' % (eigs[:dim]@np.conjugate(eigs[:dim])/norm))
    return eigv[:, :dim].T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim", type=int)
    # parser.add_argument("-p", "--path", type=str)
    args = parser.parse_args()
    img, label, l, w = load_data()
    nv = PCA(img, dim=args.dim)
    data = (nv@img).T.real
    with open(args.path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
