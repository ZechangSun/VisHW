import numpy as np
from os.path import exists
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def t_sne(n_components=2):
    img = np.load(r'../data/sampled_image.npy')
    img = img.reshape((-1, img.shape[1]*img.shape[2]))
    x = TSNE(n_components=n_components).fit_transform(img)
    np.save('tsne.npy', x)


def kde(x, y, X, Y, h1, h2, kernel1, kernel2):
    kx = kernel1((x-X)/h1)
    ky = kernel2((y-Y)/h2)
    p = 1/(len(X)*h1*h2)*np.sum(kx*ky)
    return p


def gaussian(x):
    return 1/(2*np.pi)**0.5*np.exp(-x*x/2)


def epanechnikov(x):
    x_ = 3/4*(1-x*x)
    x_[(x > 1) | (x < -1)] = 0
    return x_


def densitymap(data, h1, h2, kernel1, kernel2, n=1000):
    x = np.linspace(min(data[:, 0]), max(data[:, 0]), n)
    y = np.linspace(min(data[:, 1]), max(data[:, 1]), n)
    contour = np.zeros((n, n))
    if kernel1 == 'gaussian':
        func1 = gaussian
    elif kernel1 == 'epanechnikov':
        func1 = epanechnikov
    if kernel2 == 'gaussian':
        func2 = gaussian
    elif kernel2 == 'epanechnikov':
        func2 = epanechnikov
    for i in range(n):
        for j in range(n):
            contour[i][j] = kde(x[i], y[j], data[:, 0], data[:, 1], h1, h2, func1, func2)
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(12, 12))
    a = plt.contourf(X, Y, contour, cmap='Reds', alpha=1)
    plt.contour(X, Y, contour)
    plt.colorbar(a)
    plt.title('density map-%s-%s-%.1f-%.1f' % (kernel1, kernel2, h1, h2))
    plt.savefig('density map-%s-%s-%.1f-%.1f.png' % (kernel1, kernel2, h1, h2))
    plt.close()


if __name__ == "__main__":
    if not exists('tsne.npy'):
        t_sne()
    data = np.load('tsne.npy')
    label = np.load('../data/sampled_label.npy')
    f = plt.figure(figsize=(12, 12))
    for i in range(10):
        plt.scatter(data[label == i, 0], data[label == i, 1], label='%i' % i)
    plt.legend(loc='best')
    plt.title('T-SNE decomposition')
    plt.savefig('scatter.png')
    plt.close()
    densitymap(data, 3, 3, 'epanechnikov', 'epanechnikov')
    densitymap(data, 2, 3, 'epanechnikov', 'epanechnikov')
    densitymap(data, 2, 2, 'epanechnikov', 'epanechnikov')
    densitymap(data, 2, 2, 'gaussian', 'gaussian')
    densitymap(data, 1.5, 1.5, 'gaussian', 'gaussian')
    densitymap(data, 5, 5, 'gaussian', 'gaussian')
    densitymap(data, 3, 5, 'gaussian', 'epanechnikov')
    densitymap(data, 1.5, 3, 'gaussian', 'epanechnikov')
