from pyecharts import Scatter, Scatter3D
from pyecharts import Page
import pyecharts
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('img2d.csv', sep=',', names=['x', 'y'])
    pyecharts.configure(global_theme='shine')
    label = np.load('../data/sampled_label.npy')
    page = Page(page_title='PCA visualization')
    scatter2d = Scatter(title='PCA with 2 components', width=1400, height=720, title_pos='center')
    for i in range(10):
        scatter2d.add('%i' % i, data['x'][label == i], data['y'][label == i], legend_orient='vertical',
                    legend_pos='5%', legend_top='center', yaxis_pos='right', label_fomatter='{a}', is_datazoom_show=True, datazoom_type='both', label_formatter='{a}')
    page.add_chart(scatter2d)
    data3d = pd.read_csv('img3d.csv', sep=',', names=['x', 'y', 'z'])
    scatter3d = Scatter(title='PCA with 3 components', width=1400, height=720, title_pos='center')
    for i in range(10):
        t = list(data3d['z'][label == i])
        scatter3d.add('%i' % i, data3d['x'][label == i], data3d['y'][label == i], extra_data=list(data3d['z'][label == i]), is_visualmap=True, visual_type='size', visual_range_size=[5, 15], visual_range=[min(t), max(t)], legend_orient='vertical',
                    legend_pos='5%', legend_top='center', yaxis_pos='right', label_fomatter='{a}', is_datazoom_show=True, datazoom_type='both', label_formatter='{a}')
    page.add_chart(scatter3d)
    scatter3D = Scatter3D('PCA with 3 components (3D)', width=1400, height=720, title_pos='center')
    for i in range(10):
        scatter3D.add('%i'%i, data3d.values[label == i], legend_pos='5%', legend_orient='vertical', legend_top='center')
    page.add_chart(scatter3D)
    page.render('test.html')
